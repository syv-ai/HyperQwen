#!/usr/bin/env python3
"""CPU checks for explicit, historical and fallback code corpus selection."""
import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import random
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import code_corpus

SCRIPT = Path(__file__).resolve().parent / "make_long_corpus.py"
QUALITY = SCRIPT.parent / "quality_battery.py"


class LongCorpusTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.package = self.root / "active-libs/vllm"
        (self.package / "v1/core").mkdir(parents=True)
        (self.package / "__init__.py").write_text('raise RuntimeError("do not import the runtime")', encoding="utf-8")
        (self.package / "v1/core/example.py").write_text("# code corpus\n" * 200, encoding="utf-8")
        (self.root / "bench").mkdir()
        (self.root / "bench/labd_corpus.txt").write_text("frozen head\n", encoding="utf-8")
        for patch in (
            mock.patch.dict(os.environ, CODE_CORPUS_DIR="", HOME=str(self.root), USERPROFILE=str(self.root)),
            mock.patch.object(code_corpus, "CHECKOUT_SOURCE", self.root / "missing-venv"),
            mock.patch.object(importlib.util, "find_spec", return_value=mock.Mock(submodule_search_locations=[str(self.package)])),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def quality_docs(self):
        # Execute the real corpus function without running network or importing pyarrow.
        function = next(n for n in ast.parse(QUALITY.read_text(encoding="utf-8")).body
                        if isinstance(n, ast.FunctionDef) and n.name == "docs")
        table = mock.Mock()
        table.column.return_value.to_pylist.return_value = ["language corpus\n" * 200]
        pq = mock.Mock(read_table=mock.Mock(return_value=table))
        scope = dict(os=os, random=random, Path=Path, Q=str(self.root), pq=pq,
                     source_files=code_corpus.source_files)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(QUALITY), "exec"), scope)
        return scope["docs"], pq

    def test_quality_battery_keeps_code_and_reports_source(self):
        docs, _ = self.quality_docs()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            corpora = docs()
        self.assertIn("code", {lang for lang, _ in corpora})
        self.assertIn(str(self.package.resolve()), out.getvalue())
        self.assertIn("source files", out.getvalue())
        self.assertIn("windows", out.getvalue())

    def test_explicit_tree_wins_over_checkout_and_interpreter(self):
        explicit = self.root / "fixed-source"
        (explicit / "v1/core").mkdir(parents=True)
        (explicit / "v1/core/fixed.py").write_text("# fixed\n" * 200, encoding="utf-8")
        with mock.patch.dict(os.environ, CODE_CORPUS_DIR=str(explicit)), \
                mock.patch.object(code_corpus, "CHECKOUT_SOURCE", self.package), \
                mock.patch.object(importlib.util, "find_spec") as find:
            root, files = code_corpus.source_files("v1/core")
        self.assertEqual(root, explicit.resolve())
        self.assertEqual([f.name for f in files], ["fixed.py"])
        find.assert_not_called()

    def test_historical_checkout_wins_over_interpreter(self):
        with mock.patch.object(code_corpus, "CHECKOUT_SOURCE", self.package), \
                mock.patch.object(importlib.util, "find_spec") as find:
            root, _ = code_corpus.source_files("v1/core")
        self.assertEqual(root, self.package.resolve())
        find.assert_not_called()

    def test_invalid_explicit_source_never_falls_back(self):
        with mock.patch.dict(os.environ, CODE_CORPUS_DIR=str(self.root / "typo")):
            with self.assertRaisesRegex(RuntimeError, "CODE_CORPUS_DIR"):
                code_corpus.source_files("v1/core")

    def test_missing_install_fails_before_parquet_or_long_output(self):
        docs, pq = self.quality_docs()
        with mock.patch.object(importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "CODE_CORPUS_DIR"):
                docs()
            pq.read_table.assert_not_called()
            with self.assertRaisesRegex(SystemExit, "CODE_CORPUS_DIR"):
                runpy.run_path(str(SCRIPT), run_name="__main__")
        self.assertFalse((self.root / "bench/labd_corpus_long.txt").exists())

    def test_missing_core_refuses_to_drop_cohort(self):
        docs, pq = self.quality_docs()
        (self.package / "v1/core/example.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "CODE_CORPUS_DIR"):
            docs()
        pq.read_table.assert_not_called()

    def test_no_eligible_windows_or_long_files_is_an_error(self):
        (self.package / "v1/core/example.py").write_text("# too short\n", encoding="utf-8")
        docs, _ = self.quality_docs()
        with self.assertRaisesRegex(RuntimeError, "No eligible code windows"):
            docs()
        with self.assertRaisesRegex(SystemExit, "No eligible source files"):
            runpy.run_path(str(SCRIPT), run_name="__main__")

    def test_fallback_package_and_frozen_utf8_head(self):
        head = "frozen head\r\n\u00e6\u00f8\u00e5\n".encode("utf-8")
        (self.root / "bench/labd_corpus.txt").write_bytes(head)
        # A private checkout makes this independent of a real local venv.
        checkout = self.root / "checkout/bench"
        checkout.mkdir(parents=True)
        for name in ("make_long_corpus.py", "code_corpus.py"):
            shutil.copy2(SCRIPT.parent / name, checkout / name)
        result = subprocess.run([sys.executable, str(checkout / SCRIPT.name)], capture_output=True, text=True,
                                env=dict(os.environ, PYTHONPATH=str(self.package.parent)))
        self.assertEqual(result.returncode, 0, result.stderr)
        data = (self.root / "bench/labd_corpus_long.txt").read_bytes()
        self.assertTrue(data.startswith(head))
        self.assertIn((self.package / "v1/core/example.py").read_text(encoding="utf-8").encode("utf-8"), data)
        self.assertIn(str(self.package.resolve()), result.stdout)


    def test_blank_frozen_base_is_refused_without_overwriting_long_corpus(self):
        for base in ("", " \t\n" * 70000, "\n" * 200000 + "nonblank tail"):
            with self.subTest(size=len(base)):
                (self.root / "bench/labd_corpus.txt").write_text(base, encoding="utf-8")
                out = self.root / "bench/labd_corpus_long.txt"
                original = b"existing frozen long document\r\n"
                out.write_bytes(original)
                with self.assertRaisesRegex(SystemExit, "is blank.*labd_bench.py"), \
                        mock.patch.object(code_corpus, "source_files", side_effect=AssertionError("source discovery reached")):
                    runpy.run_path(str(SCRIPT), run_name="__main__")
                self.assertEqual(out.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
