#!/usr/bin/env python3
"""CPU-only checks that the 25k-document demo cannot record a missing document."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock

import harness

SCRIPT = Path(__file__).resolve().parent / "demo_capture.py"


class DemoTests(unittest.TestCase):
    def test_missing_or_blank_corpus_is_invalid_before_requests(self):
        for content in (None, " \n\t", "short real input", " " * 80000, "x" * 72499):
            with self.subTest(length=None if content is None else len(content)), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                if content is not None:
                    (root / "bench").mkdir()
                    (root / "bench/labd_corpus.txt").write_text(content, encoding="utf-8")
                with mock.patch.dict(os.environ, {"HOME": tmp, "USERPROFILE": tmp}), \
                        mock.patch.object(sys, "argv", [str(SCRIPT), "test", "--out", str(root / "out")]), \
                        mock.patch.object(harness, "stream", return_value=iter(())) as stream, \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as result:
                        runpy.run_path(str(SCRIPT), run_name="__main__")
                self.assertEqual(result.exception.code, 2)
                stream.assert_not_called()
                self.assertFalse((root / "out").exists())

    def test_real_document_is_sent_to_the_copy_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bench").mkdir()
            document = "Technical source document.\n" * 3000
            (root / "bench/labd_corpus.txt").write_text(document, encoding="utf-8")
            def stream(*args, **kwargs):
                yield {"choices": [{"delta": {"content": "ok"}}]}
                yield {"choices": [], "usage": {"prompt_tokens": 25000, "completion_tokens": 2}}
            with mock.patch.dict(os.environ, {"HOME": tmp, "USERPROFILE": tmp}), \
                    mock.patch.object(sys, "argv", [str(SCRIPT), "test", "--out", str(root / "out")]), \
                    mock.patch.object(harness, "stream", side_effect=stream) as calls, \
                    contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(SCRIPT), run_name="__main__")
            self.assertEqual(calls.call_count, 4)
            sent = calls.call_args.args[1]["messages"][0]["content"]
            self.assertIn(document[:72500], sent)
            self.assertLess(len(sent), len(document))
            self.assertTrue((root / "out/test.json").exists())


if __name__ == "__main__":
    unittest.main()
