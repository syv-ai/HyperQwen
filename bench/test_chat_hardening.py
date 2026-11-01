#!/usr/bin/env python3
"""CPU-only checks for explicit and default chat-template hardening targets."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prepare"))
import harden_chat_template as hardening


class HardeningTests(unittest.TestCase):
    def setUp(self):
        # Target selection is under test; the atomic publisher has a separate crash suite.
        writer = mock.patch.object(hardening, "write_text",
                                   side_effect=lambda p, text: Path(p).write_text(text, encoding="utf-8"))
        writer.start()
        self.addCleanup(writer.stop)

    def test_target_scope_and_deduplication(self):
        for mode in ("file", "directory", "both", "default"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                explicit = root / "external/chat_template.jinja"
                other = root / "models/other/chat_template.jinja"
                for path in (explicit, other):
                    path.parent.mkdir(parents=True)
                    path.write_text(hardening.OLD, encoding="utf-8")
                args = {"file": [str(explicit)], "directory": [str(explicit.parent)],
                        "both": [str(explicit.parent), str(explicit)], "default": []}[mode]
                out = io.StringIO()
                with mock.patch.object(hardening, "__file__", str(root / "prepare/script.py")), \
                        mock.patch.object(sys, "argv", ["script.py", *args]), contextlib.redirect_stdout(out):
                    self.assertEqual(hardening.main(), 0)
                self.assertEqual(out.getvalue().count("chat_template.jinja"), 1)
                self.assertEqual(hardening.MARKER in explicit.read_text(encoding="utf-8"), mode != "default")
                self.assertEqual(hardening.MARKER in other.read_text(encoding="utf-8"), mode == "default")

    def test_help_never_rewrites_an_earlier_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_template.jinja"
            path.write_text(hardening.OLD, encoding="utf-8")
            with mock.patch.object(sys, "argv", ["script.py", str(path), "--help"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    hardening.main()
            self.assertEqual(result.exception.code, 0)
            self.assertEqual(path.read_text(encoding="utf-8"), hardening.OLD)


    def test_missing_explicit_template_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(sys, "argv", ["script.py", str(Path(tmp) / "typo.jinja")]), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(hardening.main(), 1)
            self.assertIn("no template found", err.getvalue())

    def test_relative_and_absolute_paths_are_one_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_template.jinja"
            path.write_text(hardening.OLD, encoding="utf-8")
            relative = os.path.relpath(path)
            with mock.patch.object(sys, "argv", ["script.py", str(path), relative]), \
                    mock.patch.object(hardening, "harden", return_value="unchanged") as harden, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(hardening.main(), 0)
            harden.assert_called_once()

    def test_snapshot_directory_and_symlinked_file_are_one_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob, snapshot = root / "blob", root / "snapshot"
            blob.write_text(hardening.OLD, encoding="utf-8")
            snapshot.mkdir()
            link = snapshot / "chat_template.jinja"
            try:
                link.symlink_to(blob)
            except OSError:
                self.skipTest("file symlinks are unavailable on this host")
            with mock.patch.object(sys, "argv", ["script.py", str(link), str(snapshot)]), \
                    mock.patch.object(hardening, "harden", return_value="unchanged") as harden, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(hardening.main(), 0)
            harden.assert_called_once_with(os.path.realpath(blob))


if __name__ == "__main__":
    unittest.main()
