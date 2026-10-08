#!/usr/bin/env python3
"""CPU-only checks for bench/harness.py: the key chain matches resolve_api_key.sh, one URL convention, post(),
stream() and the /metrics readers."""
import contextlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import harness

REPO = Path(__file__).resolve().parents[1]
KEY_VARS = ("OPENAI_API_KEY", "VLLM_API_KEY")


def bash_client_key(env, repo):
    script = f'REPO="$1"; . "{REPO}/resolve_api_key.sh"; resolve_client_key; printf %s "$OPENAI_API_KEY"'
    clean = {k: v for k, v in os.environ.items() if k not in KEY_VARS}
    return subprocess.run(["bash", "-c", script, "bash", str(repo)], env={**clean, **env},
                          capture_output=True, check=True).stdout.decode()


class ClientKeyTests(unittest.TestCase):
    def test_matches_resolve_client_key(self):
        cases = [
            ({"OPENAI_API_KEY": "o", "VLLM_API_KEY": "v"}, "f\n", "o"),
            ({"VLLM_API_KEY": "v"}, "f\n", "v"),
            ({"OPENAI_API_KEY": "", "VLLM_API_KEY": "v"}, None, "v"),
            ({}, "f\n\n", "f"),
            ({}, "f \r\n", "f \r"),
            ({}, None, "EMPTY"),
            ({"VLLM_API_KEY": ""}, None, "EMPTY"),
        ]
        for env, file, want in cases:
            with self.subTest(env=env, file=file), tempfile.TemporaryDirectory() as tmp:
                if file is not None:
                    Path(tmp, "api_key.txt").write_text(file)
                with mock.patch.dict(os.environ, env), mock.patch.object(harness, "REPO", tmp):
                    for k in KEY_VARS:
                        if k not in env:
                            os.environ.pop(k, None)
                    self.assertEqual(harness.client_key(), want)
                self.assertEqual(bash_client_key(env, tmp), want)


class BaseUrlTests(unittest.TestCase):
    def test_one_convention(self):
        cases = [
            ({}, "http://127.0.0.1:18020"),
            ({"PORT": "18021"}, "http://127.0.0.1:18021"),
            ({"VLLM_API": "http://h:1/v1", "PORT": "18021"}, "http://h:1"),
            ({"VLLM_API": "http://h:1/v1/"}, "http://h:1"),
            ({"VLLM_API": "http://h:1"}, "http://h:1"),
            ({"VLLM_API": "http://h:1/proxy/v1"}, "http://h:1/proxy"),
        ]
        for env, want in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                for k in ("VLLM_API", "PORT"):
                    if k not in env:
                        os.environ.pop(k, None)
                self.assertEqual(harness.base_url(), want)


@contextlib.contextmanager
def stub(body):
    """A server on a free port that answers every GET and POST with `body`, then closes the connection.
    Yields the list of requests it saw, as {method, path, auth, ctype, body}, sets VLLM_API and the key, and
    unsets VLLM_MODEL."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def answer(self):
            n = int(self.headers["Content-Length"] or 0)
            seen.append({"method": self.command, "path": self.path, "auth": self.headers["Authorization"],
                         "ctype": self.headers["Content-Type"], "body": json.loads(self.rfile.read(n)) if n else None})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = answer

        def log_message(self, format, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with mock.patch.dict(os.environ, {"VLLM_API": f"http://127.0.0.1:{srv.server_port}/v1", "OPENAI_API_KEY": "k"}):
            os.environ.pop("VLLM_MODEL", None)
            yield seen
    finally:
        srv.shutdown()
        srv.server_close()


class PostTests(unittest.TestCase):
    def test_post_sends_key_and_json(self):
        with stub(b'{"ok": 1}') as seen:
            got = harness.post("/v1/chat/completions", {"model": "m"}, timeout=10)
        self.assertEqual(got, {"ok": 1})
        self.assertEqual(seen, [{"method": "POST", "path": "/v1/chat/completions", "auth": "Bearer k",
                                 "ctype": "application/json", "body": {"model": "m"}}])

    def test_get_request_has_no_body(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k", "PORT": "1"}):
            os.environ.pop("VLLM_API", None)
            req = harness.request("/metrics")
        self.assertEqual((req.full_url, req.get_method(), req.data), ("http://127.0.0.1:1/metrics", "GET", None))
        self.assertEqual(req.get_header("Authorization"), "Bearer k")


# /v1/models as vLLM writes it: the served name first, then any LoRA adapters.
MODELS = b'{"object": "list", "data": [{"id": "served", "root": "/models/x"}, {"id": "lora", "root": "/a"}]}'


class ModelTests(unittest.TestCase):
    def test_payload_names_vllm_model_only_when_set(self):
        with stub(b"{}") as seen:
            harness.post("/v1/completions", {"prompt": "p"}, timeout=10)
            os.environ["VLLM_MODEL"] = "m"
            harness.post("/v1/completions", {"prompt": "p"}, timeout=10)
        self.assertEqual([r["body"] for r in seen], [{"prompt": "p"}, {"model": "m", "prompt": "p"}])

    def test_model_from_server_unless_vllm_model(self):
        with stub(MODELS) as seen:
            self.assertEqual(harness.model(), "served")
            out = subprocess.run([sys.executable, str(REPO / "bench" / "harness.py"), "model"],
                                 capture_output=True, check=True, text=True).stdout
            os.environ["VLLM_MODEL"] = "m"
            self.assertEqual(harness.model(), "m")
        self.assertEqual(out, "served\n")
        self.assertEqual(seen, 2 * [{"method": "GET", "path": "/v1/models", "auth": "Bearer k", "ctype": None, "body": None}])

    def test_model_cli_without_a_name(self):
        def cli():
            r = subprocess.run([sys.executable, str(REPO / "bench" / "harness.py"), "model"], capture_output=True, text=True)
            return r.returncode, r.stdout, r.stderr.count("\n")
        for body in (b'{"data": []}', b"<html>"):
            with self.subTest(body=body), stub(body):
                self.assertEqual(cli(), (1, "", 1))
        with mock.patch.dict(os.environ, {"VLLM_API": "http://127.0.0.1:9"}):
            os.environ.pop("VLLM_MODEL", None)
            self.assertEqual(cli(), (1, "", 1))


# A keep-alive comment before and between frames, a data line with no space after the colon, and a frame
# after [DONE] that must not be read.
SSE = (b': keep-alive\n\ndata: {"choices": [{"delta": {"content": "a"}}]}\n\n: keep-alive\n\n'
       b'data:{"choices": [], "usage": {"completion_tokens": 1}}\n\ndata: [DONE]\n\ndata: {"after": 1}\n\n')


class StreamTests(unittest.TestCase):
    def test_frames_until_done(self):
        with stub(SSE) as seen:
            got = list(harness.stream("/v1/chat/completions", {"stream": True}, timeout=10))
        self.assertEqual(got, [{"choices": [{"delta": {"content": "a"}}]},
                               {"choices": [], "usage": {"completion_tokens": 1}}])
        self.assertEqual((seen[0]["path"], seen[0]["auth"], seen[0]["body"]),
                         ("/v1/chat/completions", "Bearer k", {"stream": True}))


# What vLLM's prometheus_client writes for two engines: <name>_total and <name>_created per label set, with
# the accepted counter ahead of the drafts counter, so a reader that goes by line order gets them swapped.
METRICS = b"""# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 25.0
vllm:spec_decode_num_accepted_tokens_created{engine="0",model_name="m"} 1.7e+09
vllm:spec_decode_num_accepted_tokens_total{engine="1",model_name="m"} 5.0
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 10.0
vllm:spec_decode_num_drafts_created{engine="0",model_name="m"} 1.7e+09
vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 4.0
vllm:cache_config_info{block_size="16",note="a b}"} 1.0
vllm:num_preemptions_total 3.0
vllm:num_requests_running{engine="0"} 2.0 1700000000000
"""


class MetricsTests(unittest.TestCase):
    def test_samples_keep_labels_with_spaces(self):
        with stub(METRICS) as seen:
            got = harness.samples(timeout=10)
        self.assertIn(("vllm:cache_config_info", '{block_size="16",note="a b}"}', 1.0), got)
        self.assertIn(("vllm:num_preemptions_total", "", 3.0), got)
        self.assertIn(("vllm:num_requests_running", '{engine="0"}', 2.0), got)
        self.assertEqual(len(got), 9)
        self.assertEqual((seen[0]["method"], seen[0]["path"], seen[0]["auth"]), ("GET", "/metrics", "Bearer k"))

    def test_metrics_sum_engines_and_skip_created(self):
        with stub(METRICS):
            got = harness.metrics("vllm:spec_decode_num_drafts_total", "vllm:num_preemptions_total", "vllm:absent")
        self.assertEqual(got, {"vllm:spec_decode_num_drafts_total": 14.0, "vllm:num_preemptions_total": 3.0})

    def test_spec_by_name(self):
        with stub(METRICS):
            self.assertEqual(harness.spec(), (14.0, 30.0))
            out = subprocess.run([sys.executable, str(REPO / "bench" / "harness.py"), "spec"],
                                 capture_output=True, check=True, text=True).stdout
        self.assertEqual(out, "14.0 30.0\n")

    def test_spec_cli_without_server(self):
        with mock.patch.dict(os.environ, {"VLLM_API": "http://127.0.0.1:9"}):
            r = subprocess.run([sys.executable, str(REPO / "bench" / "harness.py"), "spec"], capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout, r.stderr.count("\n")), (1, "", 1))


if __name__ == "__main__":
    unittest.main()
