#!/usr/bin/env python3
"""CPU-only check that labd_bench.py's TTFT and decode rate survive a wall clock that steps back, as the Docker
Desktop (WSL2) VM's clock does each time it is pulled back to the host's. A stub server holds the first token
0.3 s, then sends 40 tokens 25 ms apart, while labd_bench.py runs with time.time() stepped back 0.15 s every
0.2 s. The bench must report no TTFT shorter than the hold and no decode rate faster than the stub sent tokens.

  python bench/test_labd_bench_clock.py
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

BENCH = Path(__file__).resolve().parent / "labd_bench.py"
HOLD, N, GAP = 0.3, 40, 0.025
STEPPED = """import os, runpy, sys, time
wall, start = time.time, time.time()
time.time = lambda: wall() - 0.15 * ((wall() - start) // 0.2)
sys.path.insert(0, os.path.dirname(sys.argv[1]))
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


class Stub(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        stream = json.loads(self.rfile.read(int(self.headers["Content-Length"]))).get("stream")
        self.send_response(200)
        self.end_headers()
        if not stream:
            self.wfile.write(b'{"choices": [{"message": {"content": "ok"}}]}')
            return
        time.sleep(HOLD)
        for i in range(N):
            time.sleep(GAP if i else 0)
            self.wfile.write(b'data: {"choices": [{"delta": {"content": "a"}}]}\n\n')
            self.wfile.flush()
        self.wfile.write(b'data: {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": %d}}\n\n' % N)
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, format, *args):
        pass


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
with tempfile.TemporaryDirectory() as tmp:
    corpus = Path(tmp) / "corpus.txt"
    corpus.write_text("x" * 4000)
    env = {k: v for k, v in os.environ.items() if k not in ("VLLM_API", "VLLM_MODEL")}
    p = subprocess.run([sys.executable, "-c", STEPPED, str(BENCH), "clock", "--ctx", "1000",
                        "--corpus", str(corpus), "--tasks", "summary"],
                       env={**env, "HOME": tmp, "PORT": str(srv.server_port), "OPENAI_API_KEY": "k"},
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    name, out, steps, tps, decode, e2e, ttft = json.loads(
        (Path(tmp) / "bench" / "results" / "labd_clock.json").read_text())["rows"][0]
srv.shutdown()

assert out == N, p.stdout
assert ttft >= HOLD, f"TTFT {ttft:.2f}s is shorter than the {HOLD}s the stub held the first token:\n{p.stdout}"
# 20% covers a first frame read late on a busy runner. A stepped clock reads 2.5x to 4x.
assert decode <= 1.2 / GAP, f"decode {decode:.1f} tok/s is faster than the stub's {1 / GAP:.0f} tok/s:\n{p.stdout}"
print("ok")
