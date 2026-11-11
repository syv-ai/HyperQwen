#!/usr/bin/env python3
"""CPU tests for saved greedy baselines; no model/server dependencies."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_greedy_repeatability.py"
spec = importlib.util.spec_from_file_location("greedy", SCRIPT)
greedy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(greedy)


class GreedyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.baseline = Path(self.tmp.name) / "saved.json"
        self.calls = 0

    def serve(self, request, timeout):
        self.calls += 1
        if request.full_url.endswith("/v1/models"):
            data = {"data": [{"id": "fixture"}]}
        else:
            data = {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1}}
        return io.BytesIO(json.dumps(data).encode())

    def run_cli(self, *args):
        with patch.object(sys, "argv", [str(SCRIPT), *args]), \
             patch.object(greedy.urllib.request, "urlopen", side_effect=self.serve), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                return greedy.main()
            except SystemExit as exc:
                return exc.code

    def records(self):
        return {name: {"text": "\u241eanswer", "tokens": 1, "finish": "stop"} for name in greedy.PROMPTS}

    def test_empty_partial_or_wrong_shape_refused_before_network(self):
        for baseline in ({}, [], {"short-fact": {"text": "answer"}},
                         {name: {"text": 42} for name in greedy.PROMPTS},
                         {name: "answer" for name in greedy.PROMPTS}):
            with self.subTest(baseline=baseline):
                self.baseline.write_text(json.dumps(baseline), encoding="utf-8")
                self.calls = 0
                self.assertEqual(self.run_cli("--compare", str(self.baseline), "--rounds", "1"), 2)
                self.assertEqual(self.calls, 0)

    def test_missing_or_invalid_json_refused_before_network(self):
        self.assertEqual(self.run_cli("--compare", str(self.baseline)), 2)
        self.baseline.write_text("{", encoding="utf-8")
        self.assertEqual(self.run_cli("--compare", str(self.baseline)), 2)
        self.assertEqual(self.calls, 0)

    def test_positive_counts_required(self):
        self.baseline.write_text(json.dumps(self.records()), encoding="utf-8")
        for flag in ("--rounds", "--max-tokens"):
            for count in ("0", "-1"):
                with self.subTest(flag=flag, count=count):
                    self.assertEqual(self.run_cli("--compare", str(self.baseline), flag, count), 2)
        self.assertEqual(self.calls, 0)

    def test_full_baseline_matches_and_checks_each_prompt(self):
        self.baseline.write_text(json.dumps(self.records()), encoding="utf-8")
        self.assertEqual(self.run_cli("--compare", str(self.baseline), "--rounds", "1"), 0)
        self.assertEqual(self.calls, 1 + len(greedy.PROMPTS))

    def test_changed_saved_text_is_a_difference(self):
        records = self.records()
        records["code"]["text"] = "changed"
        self.baseline.write_text(json.dumps(records), encoding="utf-8")
        self.assertEqual(self.run_cli("--compare", str(self.baseline), "--rounds", "1"), 1)

    def test_no_baseline_requires_two_rounds_and_saves_compatible_records(self):
        self.assertEqual(self.run_cli("--rounds", "1"), 2)
        self.assertEqual(self.run_cli("--rounds", "2", "--save", str(self.baseline)), 0)
        self.assertEqual(json.loads(self.baseline.read_text()), self.records())


if __name__ == "__main__":
    unittest.main()
