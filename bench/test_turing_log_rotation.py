#!/usr/bin/env python3
"""Dashboard log lifetime regressions, using real temporary files."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("turing_metrics", Path(__file__).resolve().parents[1]/"scripts/turing_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)

DONE = "[hyperqwen] done: 2 tokens in 1 s (2.0 tok/s) (stop, cancel=False) (req chatcmpl-new)\n"


class LogRotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"server.log"
        self.source = metrics.Source("http://127.0.0.1", "", str(self.path))

    def test_appended_partial_lines_complete_once(self):
        self.path.write_text(DONE[:40])
        self.source.read_log()
        self.assertFalse(self.source.reqs)
        with self.path.open("a") as log:
            log.write(DONE[40:])
        self.source.read_log()
        self.source.read_log()
        self.assertEqual(len(self.source.reqs), 1)

    def test_larger_replacement_read_from_start(self):
        self.path.write_text("old log\n")
        self.source.read_log()
        replacement = self.path.with_suffix(".new")
        replacement.write_text(DONE + "padding\n" * 10)
        replacement.replace(self.path)
        self.source.read_log()
        self.assertEqual(len(self.source.reqs), 1)

    def test_truncation_discards_pending_fragment(self):
        self.path.write_text("[hyperqwen] done: 99 tokens in 1 s (2.0 tok/s) (stop, cancel=False) (req chatcmpl-old)" + " padding" * 20)
        self.source.read_log()
        self.path.write_text(DONE)
        self.source.read_log()
        self.assertEqual(len(self.source.reqs), 1)
        self.assertEqual(self.source.reqs[0]["output_tokens"], 2)

    def test_missing_file_can_appear_later(self):
        self.source.read_log()
        self.path.write_text(DONE)
        self.source.read_log()
        self.assertEqual(len(self.source.reqs), 1)


if __name__ == "__main__":
    unittest.main()
