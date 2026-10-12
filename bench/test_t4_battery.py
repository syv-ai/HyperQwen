"""CPU test for bench/t4_battery.py: the --dry-run plan, the parsers, the restart gate, the verdict, and one
end-to-end run against a stub launcher and stub bench scripts in a temp copy of the tree. No GPU, no model.

  python bench/test_t4_battery.py
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import t4_battery as t4  # noqa: E402

FAILS = 0


def check(cond, what):
    global FAILS
    print(("  PASS  " if cond else "  FAIL  ") + what)
    FAILS += not cond


LABD = """LABD t4-int8-90k qa       prompt= 90009 out= 512 steps=  261 tok/step= 1.96 decode=  33.8 e2e=   4.1 ttft=120.70s
LABD t4-int8-90k summary  prompt= 90009 out= 512 steps=  261 tok/step= 1.96 decode=  33.9 e2e=  20.1 ttft= 5.43s
LABD t4-int8-90k TOTAL out=1024 steps=522 tok/step=1.96 decode=33.8 tok/s
"""
QUALITY = ("t4-int8 PPL {'en': (10.7655, 12000), 'da': (10.9101, 12000), 'code': (3.6072, 9000), "
           "'all': (8.4078, 33000)} 90s\nt4-int8 GSM8K n=200 acc=0.955 mean_tokens=201 600s\n")
BOOT = ("INFO Using TRITON_ATTN attention backend\nINFO GPU KV cache size: 136,429 tokens\n"
        "INFO FP8 FA2: bounded BF16 prefill with persistent FP8 KV\n")


def plan_of(**kw):
    import argparse
    d = dict(arms=["int8", "fp8fa2"], cells=t4.DEFAULT_CELLS, soak_minutes=240, gsm_n=200, ctx60=53000, ctx90=85000,
             ctx149=153000, env=[], attach=None, cell_timeout=100)
    d.update(kw)
    return t4.plan(argparse.Namespace(**d))


print("== the plan")
p = plan_of()
by = {(b["arm"], b["name"]): b for b in p}
check({b["env"]["FA2_FP8KV"] for b in p if b["arm"] == "int8"} == {"0"}
      and {b["env"]["FA2_FP8KV"] for b in p if b["arm"] == "fp8fa2"} == {"1"}, "FA2_FP8KV is 0 on int8 and 1 on fp8fa2")
check(all(b["env"]["SPEC"] == "dflash2" and b["env"]["CTX"] == "long" and b["env"]["PREFIX_CACHE"] == "1" for b in p),
      "every boot is SPEC=dflash2 CTX=long PREFIX_CACHE=1")
check([b["name"] for b in p if b["arm"] == "fp8fa2"] == ["residue", "quality", "speed"], "fp8fa2 boots: residue, quality, speed")
check(("int8", "speed") in by and [c["name"] for c in by[("int8", "speed")]["cells"]] == ["d60", "d90"], "no soak on int8")
check([c["name"] for c in by[("fp8fa2", "speed")]["cells"]] == ["d60", "d90", "soak"], "soak on fp8fa2")
check(not any(c["name"] == "d149" for b in p for c in b["cells"]), "d149 is opt-in")
d = plan_of(cells=t4.CELLS, soak_minutes=7)
deep = [b for b in d if b["name"] == "deep"]
check(len(deep) == 2 and deep[0]["env"]["MAX_LEN"] == "150000" and deep[0]["env"]["KV_MEM"] == "6100000000"
      and "EMBED_UVA" not in deep[0]["env"], "d149 boots MAX_LEN=150000 KV_MEM=6100000000, no EMBED_UVA")
check(["--minutes", "7"] == [c for b in d for c in b["cells"] if c["name"] == "soak"][0]["argv"][-2:], "--soak-minutes reaches labd_soak")
check([c["argv"][1:] for c in by[("int8", "residue")]["cells"]] == [["bench/residue_sweep.py", "t4-int8"]], "residue argv")
check(by[("int8", "quality")]["cells"][0]["argv"][1:] == ["bench/quality_battery.py", "t4-int8", "--gsm-n", "200"], "quality argv")
check(by[("int8", "speed")]["cells"][0]["argv"][2:5] == ["t4-int8-60k", "--ctx", "53000"], "d60 argv")
check(plan_of(attach="fp8fa2", arms=["fp8fa2"])[0]["name"] == "attached", "--attach is one boot-less group")
os.environ["T4_149K_KV_MEM"] = "7000000000"
check([b for b in plan_of(cells=["d149"]) if b["arm"] == "int8"][0]["env"]["KV_MEM"] == "7000000000", "T4_149K_KV_MEM overrides")
del os.environ["T4_149K_KV_MEM"]

print("== --dry-run starts nothing")
T = tempfile.mkdtemp()
r = subprocess.run([sys.executable, f"{HERE}/t4_battery.py", "--dry-run", "--out", f"{T}/o", "--cells", "residue,d90,soak"],
                   capture_output=True, text=True)
check(r.returncode == 0 and "FA2_FP8KV=0" in r.stdout and "FA2_FP8KV=1" in r.stdout and "labd_soak.py --minutes 240" in r.stdout,
      "prints both arms' env and commands, exit 0")
check(not os.path.exists(f"{T}/o"), "creates no output directory")
shutil.rmtree(T)

print("== parsers")
m = t4.parse_labd(LABD)
check(m["decode"] == 33.8 and m["prompt_tokens"] == 90009 and m["ttft_cold"] == 120.7 and m["tok_step"] == 1.96, "labd_bench TOTAL, prompt, cold TTFT")
q = t4.parse_quality(QUALITY)
check(q["gsm8k"] == 0.955 and q["ppl_all"] == 8.4078 and q["ppl"]["da"] == 10.9101, "quality_battery GSM8K and PPL")
check(t4.parse_residue("[x] DONE: 0 broken of 128 residues\n") == {"broken": 0, "of": 128}, "residue_sweep DONE line")
check(t4.parse_soak("soak: 117 rounds x 4 requests, tokens/step=2.98, OK\n")["ok"], "soak OK line")
b = t4.parse_boot_log(BOOT)
check(b["pool_tokens"] == 136429 and b["fa2_fp8_line"] and len(b["backend_lines"]) == 2, "boot log: pool, backend lines")
check(t4.parse_labd("") == {} and t4.parse_quality("garbage") == {}, "empty output parses to nothing")

check(t4.judge("residue", 1, {"broken": 2, "of": 128})[1] == ["2 of 128 residues broken"]
      and t4.judge("residue", 0, {"broken": 0, "of": 128})[0] == "pass", "judge: residue pass and broken")
print("== the arm check on PRINT_ARGV")
I8 = ["--attention-backend", "TRITON_ATTN", "--kv-cache-dtype", "int8_per_token_head", "--speculative-config", "{}"]
F8 = ["--attention-backend", "FLASH_ATTN", "--kv-cache-dtype", "fp8", "--speculative-config",
      '{"method":"dflash","attention_backend":"FLASH_ATTN"}']
check(t4.check_argv("int8", I8) == [] and t4.check_argv("fp8fa2", F8) == [], "each arm's argv passes its own check")
check(len(t4.check_argv("fp8fa2", I8)) == 3, "a launcher that ignored FA2_FP8KV=1 is refused")
check(len(t4.check_argv("int8", F8)) == 2, "int8 arm on the fp8 argv is refused")

print("== the restart gate")


def fake_proc(root, procs):
    """procs: {pid: (ppid, starttime, cmdline)}; a /proc with comm names that hold spaces and parens."""
    for pid, (ppid, start, cmd) in procs.items():
        os.makedirs(f"{root}/{pid}")
        open(f"{root}/{pid}/stat", "w").write(f"{pid} (VLLM::Engine Core) S {ppid} " + " ".join(["0"] * 17) + f" {start} 0\n")
        open(f"{root}/{pid}/cmdline", "wb").write(cmd.replace(" ", "\0").encode())
    os.makedirs(f"{root}/self", exist_ok=True)


P = tempfile.mkdtemp()
fake_proc(P, {100: (1, 5000, "vllm serve m"), 101: (100, 5100, "VLLM::EngineCore"), 102: (100, 5101, "python tokenizer"),
              200: (1, 6000, "VLLM::EngineCore")})
base = t4.proc_snapshot(100, P)
check(base == {"server": [100, 5000], "engines": [[101, 5100]]}, "snapshot: server and its EngineCore child only")
check(t4.restart_gate(base, t4.proc_snapshot(100, P), "INFO fine\n") == [], "a clean cell passes")
shutil.rmtree(f"{P}/100")
check(any("server process changed" in w for w in t4.restart_gate(base, t4.proc_snapshot(100, P), "")), "the server PID gone fails the cell")
fake_proc(P, {100: (1, 5999, "vllm serve m")})  # the PID came back, a different process
check(any("server process changed" in w for w in t4.restart_gate(base, t4.proc_snapshot(100, P), "")), "a recycled server PID fails (start time)")
shutil.rmtree(f"{P}/100")
shutil.rmtree(f"{P}/101")
fake_proc(P, {100: (1, 5000, "vllm serve m"), 111: (100, 7000, "VLLM::EngineCore")})
w = t4.restart_gate(base, t4.proc_snapshot(100, P), "")
check(len(w) == 1 and "EngineCore processes changed" in w[0], "same server, a new EngineCore: fails")
shutil.rmtree(P)
same = {"server": [1, 2], "engines": [[3, 4]]}
check(any("EngineCore death" in w for w in t4.restart_gate(
    same, same, "INFO ok\nERROR Engine core proc EngineCore_0 died unexpectedly, shutting down client.\n")), "a death line fails")
check(any("EngineCore death" in w for w in t4.restart_gate(same, same, "vllm.v1.engine.exceptions.EngineDeadError\n")), "EngineDeadError fails")
dk = {"container": "c", "restart_count": 0, "started_at": "t0", "running": True}
check(t4.restart_gate(dk, dict(dk), "") == [] and t4.restart_gate(dk, {**dk, "restart_count": 1}, "")
      and t4.restart_gate(dk, {**dk, "running": False}, ""), "docker: RestartCount change or a stopped container fails")


def cell(name, status="pass", metrics=None, gate=()):
    return {"name": name, "status": status, "reasons": [], "gate_reasons": list(gate), "metrics": metrics or {}, "rc": 0, "secs": 1}


def clean(d90_fp8=40.9, gsm_fp8=0.960, ppl_fp8=8.4092, **over):
    def lab(x, n):
        return {"decode": x, "prompt_tokens": n, "tok_step": 1.96, "ttft_cold": 70.0}
    res = {}
    for arm, d90, gsm, ppl in (("int8", 33.8, 0.955, 8.4078), ("fp8fa2", d90_fp8, gsm_fp8, ppl_fp8)):
        res[arm] = {"boots": {"speed": {"pool_tokens": 136429 if arm == "int8" else 140902}}, "cells": {
            "residue": cell("residue", metrics={"broken": 0, "of": 128}),
            "quality": cell("quality", metrics={"gsm8k": gsm, "ppl_all": ppl}),
            "d60": cell("d60", "info", lab(43.9 if arm == "fp8fa2" else 42.7, 60275)),
            "d90": cell("d90", "info", lab(d90, 90009))}}
    res["fp8fa2"]["cells"]["soak"] = cell("soak", metrics={"rounds": 117, "tok_step": 2.98, "ok": True})
    for k, v in over.items():
        arm, c = k.split("__")
        res[arm]["cells"][c] = v
    return res


print("== the verdict")
ARMS2, CELLS = ["int8", "fp8fa2"], t4.DEFAULT_CELLS
g, v = t4.evaluate(clean(), ARMS2, CELLS)
check(v == "PASS" and all(x["status"] == "pass" for x in g), "a clean fixture passes every bar")
check(t4.evaluate(clean(d90_fp8=33.7), ARMS2, CELLS)[1] == "FAIL", "fp8fa2 slower than int8 at 90k fails")
check(t4.evaluate(clean(d90_fp8=33.8), ARMS2, CELLS)[1] == "PASS", "equal decode at 90k passes")
check(t4.evaluate(clean(gsm_fp8=0.944), ARMS2, CELLS)[1] == "FAIL", "GSM8K more than 1.0 pt below int8 fails")
check(t4.evaluate(clean(ppl_fp8=8.4078 * 1.021), ARMS2, CELLS)[1] == "FAIL", "PPL more than 2% above int8 fails")
check(t4.evaluate(clean(fp8fa2__residue=cell("residue", "fail", {"broken": 1, "of": 128})), ARMS2, CELLS)[1] == "FAIL", "a broken residue fails")
check(t4.evaluate(clean(int8__d60=cell("d60", "fail", {}, ["server process changed: [1, 2] -> None"])), ARMS2, CELLS)[1] == "FAIL",
      "a restart in any cell fails the arm")
r = clean()
del r["fp8fa2"]["cells"]["soak"]
check(t4.evaluate(r, ARMS2, CELLS)[1] == "INCOMPLETE", "a cell that never ran is INCOMPLETE")
r1 = {"fp8fa2": clean()["fp8fa2"]}
check(t4.evaluate(r1, ["fp8fa2"], CELLS)[1] == "PARTIAL", "one arm: cross-arm bars n/a, PARTIAL")
md = t4.render_md({"verdict": v, "meta": {"git_sha": "abc", "git_dirty": False, "host": "native", "gpu": "RTX 3090",
                                         "vllm_pin": "0.31.0", "fa2_pin": {}, "arms": ARMS2}, "gates": g, "arms": clean()})
check("+21.0%" in md and "| pool (tokens) | 136429 | 140902 | +3.3% |" in md and "PASS" in md.splitlines()[0], "markdown: decode delta and pool")

print("== end to end against stubs")
T = tempfile.mkdtemp()
for d in ("bench", "single-user", "docker"):
    os.makedirs(f"{T}/{d}")
for f in ("t4_battery.py", "harness.py"):
    shutil.copy(f"{HERE}/{f}", f"{T}/bench/{f}")
shutil.copy(f"{REPO}/docker/requirements.txt", f"{T}/docker/requirements.txt")
os.makedirs(f"{T}/bench/quality-data")
os.makedirs(f"{T}/home/bench")
open(f"{T}/home/bench/labd_corpus.txt", "w").close()
open(f"{T}/home/bench/labd_corpus_long.txt", "w").close()
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
PORT = str(sock.getsockname()[1])
sock.close()
open(f"{T}/single-user/start_qwen.sh", "w").write(f"""#!/bin/bash
# stub launcher: PRINT_ARGV=1 prints the arm's argv, else serves /health on $PORT as this very process
if [ "$FA2_FP8KV" = 1 ]; then A="--attention-backend FLASH_ATTN --kv-cache-dtype fp8 --speculative-config"; C='{{"attention_backend":"FLASH_ATTN"}}'
else A="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head --speculative-config"; C='{{}}'; fi
if [ "${{PRINT_ARGV:-0}}" = 1 ]; then printf '%s\\n' $A "$C"; exit 0; fi
echo "INFO Using FLASH_ATTN attention backend"; echo "INFO GPU KV cache size: 140,902 tokens"
echo $$ > "{T}/server.pid"
exec python3 -c "
import http.server
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(s): s.send_response(200); s.end_headers()
    def log_message(*a): pass
http.server.HTTPServer(('127.0.0.1', $PORT), H).serve_forever()"
""")
STUBS = {"quality_battery.py": "print(%r)" % QUALITY, "labd_bench.py": "print(%r)" % LABD,
         "residue_sweep.py": "print('[x] DONE: 0 broken of 128 residues')",
         "labd_soak.py": "print('soak: 3 rounds x 4 requests, tokens/step=2.9, OK')"}
for n, body in STUBS.items():
    open(f"{T}/bench/{n}", "w").write(body + "\n")
env = {**os.environ, "HOME": f"{T}/home", "PORT": PORT, "PY": sys.executable}


def e2e(out, *argv, **extra):
    r = subprocess.run([sys.executable, f"{T}/bench/t4_battery.py", "--out", f"{T}/{out}", "--boot-timeout", "60",
                        "--soak-minutes", "1", *argv], capture_output=True, text=True, env={**env, **extra}, cwd=T)
    return r, json.load(open(f"{T}/{out}/summary.json")) if os.path.exists(f"{T}/{out}/summary.json") else None


r, s = e2e("clean", "--cells", "residue,d90,soak")
check(r.returncode == 0 and s["verdict"] == "PASS", "clean stub run: PASS, exit 0 (%s)" % (r.stdout + r.stderr)[-300:])
check(os.path.exists(f"{T}/clean/summary.md") and os.path.exists(f"{T}/clean/fp8fa2/speed.server.log")
      and os.path.exists(f"{T}/clean/int8/residue.argv.txt"), "writes summary.md, the server log and the argv")
check(s["arms"]["fp8fa2"]["boots"]["speed"]["pool_tokens"] == 140902 and "soak" in s["arms"]["fp8fa2"]["cells"]
      and "soak" not in s["arms"]["int8"]["cells"], "boot log pool recorded, soak on fp8fa2 only")
check(subprocess.run(["pgrep", "-f", f"HTTPServer\\(\\('127.0.0.1', {PORT}"], capture_output=True).returncode == 1, "every stub server is torn down")

# a cell that kills the server: the cell and the arm fail, the rest of that boot is skipped
open(f"{T}/bench/labd_bench.py", "w").write("import os, signal\nos.kill(int(open(%r).read()), signal.SIGKILL)\n"
                                             "print(%r)\n" % (f"{T}/server.pid", LABD))
r, s = e2e("dead", "--cells", "d60,d90", "--arms", "int8")
c = s["arms"]["int8"]["cells"]
check(r.returncode == 1 and s["verdict"] == "FAIL" and c["d60"]["status"] == "fail"
      and any("server process changed" in w for w in c["d60"]["gate_reasons"]), "a server killed mid-cell fails the cell and the arm (exit 1)")
check(c["d90"]["status"] == "fail" and "not run" in c["d90"]["reasons"][0], "the next cell on that boot is skipped, not run")
open(f"{T}/bench/labd_bench.py", "w").write("print(%r)\n" % LABD)

# a launcher that ignores FA2_FP8KV is refused before it boots
s0 = open(f"{T}/single-user/start_qwen.sh").read()
open(f"{T}/single-user/start_qwen.sh", "w").write(s0.replace('if [ "$FA2_FP8KV" = 1 ]', 'if false'))
r, s = e2e("knob", "--cells", "residue", "--arms", "fp8fa2")
check(r.returncode == 1 and "--attention-backend" in s["arms"]["fp8fa2"]["cells"]["residue"]["reasons"][0]
      and not os.path.exists(f"{T}/knob/fp8fa2/residue.server.log"), "a launcher without the knob: fp8fa2 refused, nothing booted")
shutil.rmtree(T)

print()
print("all checks passed" if not FAILS else f"{FAILS} checks FAILED")
sys.exit(1 if FAILS else 0)
