#!/usr/bin/env python3
"""CPU regressions for quality-battery arguments and actual GSM8K sample counts."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name("quality_battery.py")


class QualityBatteryTests(unittest.TestCase):
    def cli(self, *args):
        # Help and invalid arguments must work without site packages or a server.
        return subprocess.run([sys.executable, "-S", str(SCRIPT), *args],
                              capture_output=True, text=True, timeout=10)

    def test_help_explains_tag_without_importing_optional_dependencies(self):
        result = self.cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("result_<tag>.json", result.stdout)
        self.assertIn("--gsm-only", result.stdout)
        self.assertIn("mycard", result.stdout)

    def test_invalid_arguments_are_refused_before_dependencies_or_requests(self):
        cases = [[], ["--gsm-only"], ["mycard", "--gsm-n", "0"],
                 ["mycard", "--gsm-n", "-1"], ["mycard", "--gsm-n", "many"],
                 ["mycard", "--gsm-n"], ["mycard", "--gsm-onyl"],
                 ["mycard", "--gsm-only", "--ppl-only"]]
        for args in cases:
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("usage:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def run_gsm(self, directory, rows, *extra):
        parquet = ModuleType("pyarrow.parquet")
        table = mock.Mock()
        table.column.side_effect = lambda name: mock.Mock(to_pylist=lambda: [r[name] for r in rows])
        parquet.read_table = mock.Mock(return_value=table)
        arrow = ModuleType("pyarrow")
        arrow.parquet = parquet
        harness = ModuleType("harness")
        harness.post = mock.Mock(return_value={"choices": [{"message": {"content": "Final answer: 4"}}],
                                              "usage": {"completion_tokens": 8}})
        output = io.StringIO()
        with mock.patch.dict(sys.modules, {"pyarrow": arrow, "pyarrow.parquet": parquet, "harness": harness}), \
                mock.patch.dict(os.environ, QUALITY_DATA=str(directory)), \
                mock.patch.object(sys, "argv", [str(SCRIPT), "mycard", "--gsm-only", *extra]), \
                contextlib.redirect_stdout(output):
            runpy.run_path(str(SCRIPT), run_name="__main__")
        return json.loads((directory / "result_mycard.json").read_text()), output.getvalue(), harness.post

    def test_gsm_reports_the_number_actually_evaluated(self):
        rows = [{"question": "2+2?", "answer": "#### 4"}] * 2
        for extra, expected in [((), 2), (("--gsm-n", "1"), 1), (("--gsm-n", "9999"), 2)]:
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                result, output, post = self.run_gsm(Path(tmp), rows, *extra)
                self.assertEqual(result["gsm8k"], {"n": expected, "acc": 1.0, "mean_tokens": 8})
                self.assertIn(f"mycard GSM8K n={expected} acc=1.000", output)
                self.assertEqual(post.call_count, expected)
                self.assertNotIn("ppl", result)

    def test_empty_gsm_dataset_is_invalid_and_writes_no_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaises(SystemExit) as exc, contextlib.redirect_stderr(io.StringIO()):
                self.run_gsm(directory, [])
            self.assertEqual(exc.exception.code, 2)
            self.assertFalse((directory / "result_mycard.json").exists())


if __name__ == "__main__":
    unittest.main()
