#!/usr/bin/env python3
"""Exercise the exported terminal module and dashboard together, without vLLM/GPU."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("metrics", ROOT/"scripts/turing_metrics.py")
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class RequestIdentityTests(unittest.TestCase):
    def terminal(self, disabled):
        # This topic creates the complete terminal module, so its exported added
        # lines are the exact source installed by apply.sh, not a reimplementation.
        text = (ROOT/"patches/terminal-request-progress.patch").read_text(encoding="utf-8")
        section = text.split("+++ b/v1/metrics/terminal_progress.py\n", 1)[1].split("\ndiff --git", 1)[0]
        source = "\n".join(line[1:] for line in section.splitlines() if line.startswith("+"))
        module = ModuleType("terminal_fixture")
        envs = SimpleNamespace(VLLM_TERMINAL_PROGRESS=True, VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=disabled)
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(envs=envs)}):
            exec(compile(source, "terminal_progress.py", "exec"), module.__dict__)
        return module

    def check_ids(self, disabled):
        terminal = self.terminal(disabled)
        dashboard = metrics.Source("http://127.0.0.1", "", "unused")
        ids = ["chatcmpl-batch-part-a", "chatcmpl-batch-part-b", "chatcmpl-job-deadbeef"]
        for external in ids:
            internal = external if disabled else external + "-abc01234"
            req = SimpleNamespace(request_id=internal, num_prompt_tokens=100)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                terminal.reading(req, 50, start=1)
            self.assertIn("(req " + external + ")", log.getvalue())
            for line in log.getvalue().splitlines():
                dashboard.parse(line, 1)
        self.assertEqual(set(dashboard.cur), set(ids))
        for external in ids:
            dashboard.parse(f"[hyperqwen] prefill: 100 new tokens in 1 s (100 tok/s), cached 0 of 100 (req {external})", 2)
        self.assertEqual(set(dashboard.cur), set(ids))
        dashboard.parse(f"[hyperqwen] done: 2 tokens in 1 s (2 tok/s) (stop, cancel=False) (req {ids[0]})", 3)
        self.assertEqual(set(dashboard.cur), set(ids[1:]))
        self.assertEqual(dashboard.reqs[0]["prompt_tokens"], 100)

    def test_randomized_internal_ids_join_their_external_phases(self):
        self.check_ids(False)

    def test_disabled_randomization_preserves_entire_caller_id(self):
        self.check_ids(True)


if __name__ == "__main__":
    unittest.main()
