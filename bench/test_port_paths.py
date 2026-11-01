#!/usr/bin/env python3
"""CPU-only port-tool checks using a real git checkout whose path contains spaces."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class PortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hq-port-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fork = self.root / "vllm checkout"
        self.fork.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "test")
        self.git("config", "user.email", "test@example.invalid")
        (self.fork / "vllm").mkdir()
        path = self.fork / "vllm/example.py"
        path.write_text("x = 1\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        self.git("tag", "v0.30.0")
        self.git("tag", "v0.31.0")
        path.write_text("x = 2\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "[qwen38] path-test", "-m", "Body copied from the topic commit.")
        self.git("branch", "topics")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.fork), *args], check=True, capture_output=True, text=True)

    def run_script(self, script, *args, env=None):
        return subprocess.run(["bash", str(REPO / "scripts" / script), self.fork.as_posix(), *args],
                              capture_output=True, text=True, timeout=30, env=env)

    def test_export(self):
        out = self.root / "export.patch"
        result = self.run_script("export-patch.sh", "topics", out.as_posix())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("+x = 2", out.read_text(encoding="utf-8"))
        self.assertIn("Body copied from the topic commit.", out.read_text(encoding="utf-8"))

    def test_drift(self):
        result = self.run_script("port-drift.sh", "v0.31.0", "topics", "v0.30.0", "topics")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 topics identical", result.stdout)

    def test_triage(self):
        result = self.run_script("port-triage.sh", "v0.30.0", "v0.31.0", "topics")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 topics: 1 clean", result.stdout)
        self.assertEqual(self.git("rev-parse", "topics").stdout, self.git("rev-parse", "HEAD").stdout)
        self.assertEqual(len(self.git("worktree", "list").stdout.splitlines()), 1)

    def test_drift_reports_changed_and_one_sided_files(self):
        for branch, base, x, name in (("new", "v0.31.0", 2, "n"), ("old", "v0.30.0", 3, "o")):
            self.git("checkout", "-qb", branch, base)
            (self.fork / "vllm/example.py").write_text(f"x = {x}\n", encoding="utf-8")
            (self.fork / f"vllm/{name}.py").write_text(f"{name} = 1\n", encoding="utf-8")
            self.git("add", ".")
            self.git("commit", "-qm", "[qwen38] path-test")
        result = self.run_script("port-drift.sh", "v0.31.0", "new", "v0.30.0", "old")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DRIFT path-test -> example.py:2 n.py:1 o.py:1", result.stdout)

    def test_failed_worktree_add_also_cleans_temp_directory(self):
        temp_root = self.root / "temp space"
        temp_root.mkdir()
        # A regular file blocks git's worktree metadata directory on both OSes.
        (self.fork / ".git/worktrees").write_text("not a directory", encoding="utf-8")
        result = self.run_script("port-triage.sh", "v0.30.0", "v0.31.0", "topics",
                                 env=dict(os.environ, TMPDIR=temp_root.as_posix()))
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(list(temp_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
