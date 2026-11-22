#!/usr/bin/env python3
"""Exercise the EKV stress client's summary and exit status without a model or GPU."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import io
import json
import os
import runpy
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("ekv_stress.py")
USAGE = "vllm:kv_cache_usage_perc"


class Output(io.StringIO):
    def reconfigure(self, **kwargs):
        pass


class StressReportTests(unittest.TestCase):
    def measure(self, metric, memory, *, completion_status=200):
        class Handler(BaseHTTPRequestHandler):
            def answer(self, status, body):
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.answer(401 if metric is None else 200, (metric or "").encode())

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                data = {"choices": [{"message": {"content": "answer"}}], "usage": {"prompt_tokens": 3}}
                self.answer(completion_status, json.dumps(data).encode())

            def log_message(self, *_):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            output = Output()
            argv = [str(SCRIPT), "fixture", str(server.server_port)]
            gpu = subprocess.CompletedProcess(["nvidia-smi"], int(memory is None), stdout=memory or "")
            try:
                with patch.object(sys, "argv", argv), patch.object(sys, "stdout", output), \
                     patch.dict(os.environ, {"VLLM_API_KEY": "fixture", "LONG_WORDS": "14", "NO_PROXY": "127.0.0.1"}), \
                     patch("subprocess.run", return_value=gpu), patch("zlib.compress", return_value=b"fixture"), \
                     self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(SCRIPT), run_name="__main__")
            finally:
                server.shutdown()
                worker.join()
        summary = next(line for line in output.getvalue().splitlines() if " SUMMARY " in line)
        self.assertIn("requests 30 ", summary)
        return result.exception.code, summary

    def test_http_success_with_no_telemetry_is_invalid(self):
        code, summary = self.measure(None, None)
        self.assertEqual(code, 2)
        self.assertIn("failed 0", summary)
        self.assertIn("peak_card_MiB unavailable peak_kv_usage unavailable", summary)

    def test_measured_zero_is_valid(self):
        code, summary = self.measure(f'{USAGE}{{engine="0"}} 0 1730000000\n', "0\n")
        self.assertEqual(code, 0)
        self.assertIn("peak_card_MiB 0 peak_kv_usage 0.000", summary)

    def test_missing_exact_metric_does_not_borrow_another_metric(self):
        code, summary = self.measure(f"{USAGE}_total 1\n{USAGE} NaN\n", "123\n")
        self.assertEqual(code, 2)
        self.assertIn("peak_card_MiB 123 peak_kv_usage unavailable", summary)

    def test_failed_requests_keep_exit_one(self):
        code, summary = self.measure(None, None, completion_status=503)
        self.assertEqual(code, 1)
        self.assertIn("failed 30", summary)
        self.assertIn("turn2_cached ['unavailable', 'unavailable']", summary)


if __name__ == "__main__":
    unittest.main()
