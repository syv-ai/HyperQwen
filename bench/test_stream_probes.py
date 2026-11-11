#!/usr/bin/env python3
"""CPU response fixtures for complete-stream measurements, without a server."""
import contextlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import runpy
import sys
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("turing_bench", ROOT/"scripts/bench_turing.py")
turing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(turing)


def sse(*chunks, done=True):
    frames = [": keep-alive\n\n"]
    frames.extend("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    if done:
        frames.append("data: [DONE]\n\n")
    return "".join(frames).encode()


TEXT = {"choices": [{"text": "token"}]}
EMPTY = {"choices": [{"text": "", "finish_reason": "stop"}]}
USAGE = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
CHAT = {"choices": [{"delta": {"reasoning_content": "thought"}}]}


class StreamProbeTests(unittest.TestCase):
    def probe(self, name, payload, args=None, wall_clock=False):
        output = io.StringIO()
        path = ROOT/"opt-170hx"/name
        clock = itertools.count(1000, -1)
        responses = iter(payload) if isinstance(payload, list) else None
        lock = threading.Lock()

        def respond(*args, **kwargs):
            with lock:
                return io.BytesIO(next(responses) if responses is not None else payload)

        with patch.object(sys, "argv", [str(path), *(args or ["10", "2", "2"])]), \
             patch("urllib.request.urlopen", side_effect=respond), \
             patch("time.time", side_effect=(lambda: next(clock)) if wall_clock else __import__("time").time), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            try:
                runpy.run_path(str(path), run_name="__main__")
                status = 0
            except SystemExit as exc:
                status = exc.code
            except (RuntimeError, ValueError, IndexError, ZeroDivisionError):
                status = 1
        return status, output.getvalue()

    def test_turing_rejects_truncated_error_or_unmeasured_streams(self):
        invalid = [sse(CHAT, USAGE, done=False), sse({"error": {"message": "failed"}}),
                   sse(CHAT), sse(USAGE), sse(CHAT, {"usage": {"completion_tokens": -1}})]
        for payload in invalid:
            with self.subTest(payload=payload), patch.object(turing, "post", return_value=io.BytesIO(payload)):
                with self.assertRaises((RuntimeError, ValueError)):
                    turing.stream_once("fixture", "test", 2)

    def test_turing_measures_reasoning_with_complete_usage(self):
        with patch.object(turing, "post", return_value=io.BytesIO(sse(CHAT, CHAT, USAGE))):
            result = turing.stream_once("fixture", "test", 2)
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["prompt_tokens"], 10)
        self.assertGreaterEqual(result["ttft_s"], 0)
        self.assertGreater(result["decode_tok_s"], 0)

    def test_cmp_probes_refuse_incomplete_error_or_missing_usage(self):
        for name in ("bench_stream_probe.py", "bench_multi_probe.py"):
            for payload in (sse(TEXT, TEXT, USAGE, done=False), sse(TEXT, TEXT),
                            sse({"error": "failed"}), sse(EMPTY, USAGE)):
                with self.subTest(name=name, payload=payload):
                    self.assertNotEqual(self.probe(name, payload)[0], 0)

    def test_cmp_probes_count_only_text_chunks(self):
        payload = sse(TEXT, TEXT, EMPTY, USAGE)
        status, text = self.probe("bench_stream_probe.py", payload)
        self.assertEqual(status, 0, text)
        self.assertIn("step_chunks=2 completion_tokens=2", text)
        status, text = self.probe("bench_multi_probe.py", payload)
        self.assertEqual(status, 0, text)
        result = json.loads(text)
        self.assertEqual(result["ok"], 2)
        self.assertEqual(result["chunks_per_stream"], 2)
        self.assertEqual(result["tok_per_chunk"], 1)

    def test_partial_concurrent_failure_returns_nonzero(self):
        status, text = self.probe("bench_multi_probe.py", [sse(TEXT, TEXT, USAGE), sse(TEXT, done=False)])
        self.assertEqual(status, 1, text)
        self.assertEqual(json.loads(text)["ok"], 1)

    def test_sse_optional_space_is_supported(self):
        payload = sse(TEXT, TEXT, USAGE).replace(b"data: ", b"data:")
        for name in ("bench_stream_probe.py", "bench_multi_probe.py"):
            with self.subTest(name=name):
                status, text = self.probe(name, payload)
                self.assertEqual(status, 0, text)

    def test_wall_clock_step_cannot_make_timings_negative(self):
        payload = sse(TEXT, TEXT, USAGE)
        status, text = self.probe("bench_stream_probe.py", payload, wall_clock=True)
        self.assertEqual(status, 0, text)
        self.assertNotIn("TTFT_ms=-", text)
        self.assertNotIn("wall_ms=-", text)
        status, text = self.probe("bench_multi_probe.py", payload, wall_clock=True)
        self.assertEqual(status, 0, text)
        self.assertGreaterEqual(json.loads(text)["ttft_min_ms"], 0)
        self.assertGreaterEqual(json.loads(text)["wall_s"], 0)

    def test_positive_counts_fail_before_network(self):
        for name in ("bench_stream_probe.py", "bench_multi_probe.py"):
            with self.subTest(name=name):
                self.assertEqual(self.probe(name, b"", ["0", "2", "1"])[0], 2)
        with patch.object(sys, "argv", ["bench_turing", "--runs", "0"]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                turing.main()
            self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
