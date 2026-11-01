#!/usr/bin/env python3
"""CPU-only check for conc_ladder.py --salt, against a stub server that reports a prefix-cache hit for any
prompt it has seen before: two runs with the same --salt send the same prompt, the second run says it was
served from the cache, and --shared, which hits the cache by design, does not.

  python bench/test_conc_ladder_salt.py
"""
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

LADDER = Path(__file__).resolve().parent / "conc_ladder.py"
HIT = "served from the prefix cache"
prompts = []


class Stub(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        prompt = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["messages"][0]["content"]
        cached = 4 if prompt in prompts else 0
        prompts.append(prompt)
        self.send_response(200)
        self.end_headers()
        usage = {"completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": cached}}
        for frame in ({"choices": [{"delta": {"content": "a"}}]}, {"choices": [{"delta": {"content": "b"}}]},
                      {"choices": [], "usage": usage}):
            self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format, *args):
        pass


def ladder(port, tmp, *flags):
    out = Path(tmp) / "ladder.json"
    env = {k: v for k, v in os.environ.items() if k not in ("VLLM_API", "VLLM_MODEL")}
    p = subprocess.run([sys.executable, str(LADDER), *flags, "--n", "1", "--reps", "1", "--out", "2",
                        "--ctx-tokens", "44", "--json", str(out)],
                       env={**env, "PORT": str(port), "OPENAI_API_KEY": "k"}, capture_output=True, text=True,
                       timeout=60)
    assert p.returncode == 0, p.stderr
    return p.stdout, json.loads(out.read_text())


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
with tempfile.TemporaryDirectory() as tmp:
    out1, j1 = ladder(srv.server_port, tmp, "--salt", "S")
    out2, j2 = ladder(srv.server_port, tmp, "--salt", "S")
    out3, j3 = ladder(srv.server_port, tmp, "--shared", "--reps", "2")
srv.shutdown()

assert prompts[0].startswith("Document 0 (revision S-n1-r0). "), prompts[0][:60]
assert prompts[1] == prompts[0], "a second run with the same --salt sends the same prompt"
assert j1["salt"] == "S" and [(r["n"], r["rep"]) for r in j1["reps"]] == [(1, 0)], j1
assert HIT not in out1 and j1["reps"][0]["cached"] == 0, "a first run reports no prefix-cache hit:\n" + out1
assert HIT in out2 and j2["reps"][0]["cached"] == 4, "a second run with the same --salt reports the hit:\n" + out2
assert j3["reps"][1]["cached"] == 4 and HIT not in out3, "--shared hits the cache by design and says nothing:\n" + out3
print("ok")
