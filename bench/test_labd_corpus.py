#!/usr/bin/env python3
"""CPU-only checks for the LABD scripts' first-run corpus, without a model or server."""
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock

import harness
from labd_corpus import load_corpus

BENCH = Path(__file__).resolve().parent
DOCUMENT = "# HyperQwen test document\n\nCUDA kernels and KV-cache commands: \u00e6\u00f8\u00e5.\n"


class ReachedServer(Exception):
    pass


class CorpusTests(unittest.TestCase):
    def run_until_request(self, script, home, repo):
        # Both scripts make a warm-up request after loading the corpus. Stop there:
        # this checks the actual prompt, without depending on timing or a live model.
        with mock.patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home)}), \
                mock.patch.object(harness, "REPO", str(repo)), \
                mock.patch.object(sys, "argv", [script, "corpus-test"]), \
                mock.patch.object(harness, "post", side_effect=ReachedServer) as post:
            with self.assertRaises(ReachedServer):
                runpy.run_path(str(BENCH / script), run_name="__main__")
        return post.call_args.args[1]["messages"][0]["content"]

    def test_first_run_uses_checkout_documents(self):
        for script in ("labd_bench.py", "labd_accept.py"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                repo = home / "renamed-checkout"
                repo.mkdir()
                (repo / "README.md").write_text(DOCUMENT, encoding="utf-8")
                prompt = self.run_until_request(script, home, repo)
                corpus = (home / "bench/labd_corpus.txt").read_text(encoding="utf-8")
                self.assertIn(DOCUMENT, prompt)
                self.assertGreaterEqual(len(corpus), 200000)
                self.assertTrue(corpus.startswith(DOCUMENT))

    def test_existing_corpus_is_reused(self):
        for script in ("labd_bench.py", "labd_accept.py"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                corpus = home / "bench/labd_corpus.txt"
                corpus.parent.mkdir()
                original = (DOCUMENT * 2000).encode("utf-8")
                corpus.write_bytes(original)
                prompt = self.run_until_request(script, home, home / "missing-checkout")
                self.assertIn(DOCUMENT, prompt)
                self.assertEqual(corpus.read_bytes(), original)

    def test_empty_documentation_does_not_publish_a_blank_corpus(self):
        for content in (None, " \n\t"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                repo = Path(tmp)
                if content is not None:
                    (repo / "README.md").write_text(content, encoding="utf-8")
                corpus = repo / "bench/corpus.txt"
                with mock.patch.object(harness, "REPO", tmp):
                    with self.assertRaisesRegex(ValueError, "no documentation found"):
                        load_corpus(corpus)
                self.assertFalse(corpus.exists())

    def test_documents_are_sorted_and_subdirectory_readmes_are_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "PATCHES.md").write_text("# Patches", encoding="utf-8")
            (repo / "README.md").write_text(DOCUMENT, encoding="utf-8")
            (repo / "a-b.md").write_text("# Sibling", encoding="utf-8")
            (repo / "a").mkdir()
            (repo / "a/README.md").write_text("# Nested", encoding="utf-8")
            (repo / "single-user").mkdir()
            (repo / "single-user/README.md").write_text("# Single user", encoding="utf-8")
            # Deliberately unordered glob results: independent of filesystem order.
            def glob(path, pattern):
                return ([repo / "README.md", repo / "a-b.md", repo / "PATCHES.md"] if pattern == "*.md"
                        else [repo / "single-user/README.md", repo / "a/README.md"])
            with mock.patch.object(harness, "REPO", tmp), mock.patch.object(Path, "glob", glob):
                corpus = load_corpus(repo / "corpus.txt")
            self.assertTrue(corpus.startswith("# Patches\n\n" + DOCUMENT + "\n\n# Sibling\n\n# Nested\n\n# Single user"))

    def test_preexisting_blank_corpus_is_refused_without_rewriting_or_requests(self):
        for script in ("labd_bench.py", "labd_accept.py"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                corpus = home / "bench/labd_corpus.txt"
                corpus.parent.mkdir()
                original = b"\n" * 200000
                corpus.write_bytes(original)
                with mock.patch.dict(os.environ, HOME=tmp, USERPROFILE=tmp), \
                        mock.patch.object(sys, "argv", [script, "blank-test"]), \
                        mock.patch.object(harness, "post", side_effect=AssertionError("server reached before corpus validation")) as post:
                    with self.assertRaisesRegex(ValueError, "is blank.*delete it"):
                        runpy.run_path(str(BENCH / script), run_name="__main__")
                post.assert_not_called()
                self.assertEqual(corpus.read_bytes(), original)

    def test_missing_long_corpus_is_not_replaced_with_short_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "labd_corpus_long.txt"
            with mock.patch.object(sys, "argv", ["labd_bench.py", "long-test", "--corpus", str(corpus)]), \
                    mock.patch.object(harness, "post", side_effect=AssertionError("server reached before corpus validation")) as post:
                with self.assertRaisesRegex(ValueError, "make_long_corpus.py"):
                    runpy.run_path(str(BENCH / "labd_bench.py"), run_name="__main__")
            post.assert_not_called()
            self.assertFalse(corpus.exists())

    def test_long_corpus_with_blank_frozen_head_is_refused_before_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "labd_corpus_long.txt"
            original = b"\n" * 200000 + DOCUMENT.encode("utf-8") * 2000
            corpus.write_bytes(original)
            with mock.patch.object(sys, "argv", ["labd_bench.py", "blank-head", "--corpus", str(corpus)]), \
                    mock.patch.object(harness, "post", side_effect=AssertionError("server reached")) as post:
                with self.assertRaisesRegex(ValueError, "blank.*make_long_corpus.py"):
                    runpy.run_path(str(BENCH / "labd_bench.py"), run_name="__main__")
            post.assert_not_called()
            self.assertEqual(corpus.read_bytes(), original)

    def test_soak_and_sweeps_refuse_blank_corpus_before_tokenizer_or_requests(self):
        for script in ("labd_soak.py", "residue_sweep.py", "bugb_sweep.py"):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                home = Path(tmp)
                corpus = home / "bench/labd_corpus.txt"
                corpus.parent.mkdir()
                original = b" \t\n" * 70000
                corpus.write_bytes(original)
                with mock.patch.dict(os.environ, HOME=tmp, USERPROFILE=tmp, CORPUS=str(corpus)), \
                        mock.patch.object(sys, "argv", [script, "blank-test"]), \
                        mock.patch.object(harness, "post", side_effect=AssertionError("server reached")) as post, \
                        mock.patch.dict(sys.modules, {"transformers": None}):
                    with self.assertRaisesRegex(ValueError, "is blank.*labd_bench.py"):
                        runpy.run_path(str(BENCH / script), run_name="__main__")
                post.assert_not_called()
                self.assertEqual(corpus.read_bytes(), original)

    def test_relative_corpus_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "README.md").write_text(DOCUMENT, encoding="utf-8")
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                with mock.patch.object(harness, "REPO", tmp):
                    self.assertIn(DOCUMENT, load_corpus("corpus.txt"))
                self.assertTrue(Path("corpus.txt").is_file())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
