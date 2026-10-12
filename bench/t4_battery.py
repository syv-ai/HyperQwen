#!/usr/bin/env python3
"""The T4 battery (#153): shipped int8 + Triton against fp8 KV on the FA2 source port, one command.

Both arms are `SPEC=dflash2 CTX=long PREFIX_CACHE=1` through single-user/start_qwen.sh:

  int8    FA2_FP8KV=0  int8_per_token_head KV on TRITON_ATTN (what ships)
  fp8fa2  FA2_FP8KV=1  fp8 KV on the FA2 port (fa2/install.sh, launcher knob from the T4 launcher PR)

Cells, per arm. Each group of cells gets its own boot, so the order below is the boot order:
  residue   cold boot, defaults       bench/residue_sweep.py t4-<arm>                      (128 residues)
  quality   roomy boot                bench/quality_battery.py t4-<arm> --gsm-n 200
  d149      MAX_LEN/KV_MEM boot       bench/labd_bench.py ... --ctx 153000  (opt-in: --cells ...,d149)
  d60 d90   cold boot, defaults       bench/labd_bench.py ... --ctx 53000 / 85000 --tasks qa,summary
  soak      same boot as d60/d90      bench/labd_soak.py --minutes N                  (fp8fa2 only)
--ctx 53000 / 85000 / 153000 read about 60k / 90k / 149k tokens on this corpus (run 4). Read `prompt=` in the
raw logs, and adjust with --ctx60 / --ctx90 / --ctx149.

Pass bars (everything else is reported, not gated):
  restart gate  Before and after every cell the server's PID and start time, and its EngineCore children's
                (or the container's RestartCount), are recorded. Any change, or an EngineCore death line in
                the server log during the cell, fails that cell and its arm (club-3090 #1358). Xid lines from
                dmesg/journalctl -k are recorded when readable and never fail a cell.
  quality       The battery has no verdict of its own. A battery that completes passes. Against int8,
                fp8fa2's GSM8K is at most 1.0 point lower and its "PPL all" at most 2% higher (§0.3 run 1's bars).
  residue       0 broken of 128 residues (residue_sweep.py exits 0).
  decode        fp8fa2 decode (labd_bench TOTAL) at ~90k >= int8's. ~60k and ~149k are reported.
  soak          labd_soak.py exits 0 (fp8fa2 only).
Exit 0: every bar passed. 1: a bar failed. 2: a bar had no data (a cell did not run). With a single arm
(--attach) the cross-arm bars are n/a, and the verdict says PARTIAL.

Env: PY (default venv/bin/python), PORT (default 18020; the launcher and every bench script use it),
T4_QUALITY_KV_MEM / T4_QUALITY_MAX_LEN (roomy quality boot, run 4's rerun: 3758096384 / 65536),
T4_149K_KV_MEM / T4_149K_MAX_LEN (6100000000 / 150000, run 4's ~149k boot, no EMBED_UVA).
--env KEY=VAL (repeatable) is added to every boot (EMBED_UVA=1, MAX_SEQS=...).

  python3 bench/t4_battery.py --dry-run                      # the plan, no GPU, nothing started
  python3 bench/t4_battery.py                                # both arms, 240-min soak (2-3 h per arm plus the soak, estimate)
  python3 bench/t4_battery.py --arms fp8fa2 --cells d90 --soak-minutes 30
  python3 bench/t4_battery.py --attach fp8fa2 --server-log qwen.log --cells d60,d90   # server already up

Output: --out (default bench/results/t4-battery-<UTC>/): summary.json and summary.md, plus per arm the raw
output of every cell, each boot's env, PRINT_ARGV argv and server log. Needs ~/bench/labd_corpus.txt and
labd_corpus_long.txt (bench/make_long_corpus.py) and bench/quality-data/, as the scripts above do.
"""
import argparse
import ast
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import harness  # noqa: E402

ARMS = {"int8": "0", "fp8fa2": "1"}
CELLS = ["residue", "quality", "d149", "d60", "d90", "soak"]  # boot order, see the docstring
DEFAULT_CELLS = [c for c in CELLS if c != "d149"]
BARS = {"gsm_drop_pt": 1.0, "ppl_rise_pct": 2.0}
CORPUS_LONG = "~/bench/labd_corpus_long.txt"
KNOB = re.compile(r"^(SPEC|CTX|PREFIX_CACHE|PORT|HOST|PY|MODEL|EXTRA_ARGS|EMBED_UVA|KV_|MAX_|DRAFT_TOKENS|GPU_UTIL|"
                  r"ASYNC_SCHED|INT8_|CHUNK|VLLM_|KVARN_|FA2_)")
SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD")
DEATH = re.compile(r"EngineDeadError|Engine core proc \S+ died|EngineCore.*(died|encountered an issue|fatal)", re.I)


# ---- the plan: shared by --dry-run and the real run ---------------------------------------------------------

def plan(a):
    """[{arm, name, env, cells: [{name, argv, timeout}]}]. env is what the script adds to the launcher."""
    py = os.environ.get("PY", "venv/bin/python")
    port = os.environ.get("PORT", "18020")
    ctx = {"d60": a.ctx60, "d90": a.ctx90, "d149": a.ctx149}

    def argv(arm, cell):
        if cell == "quality":
            return [py, "bench/quality_battery.py", f"t4-{arm}", "--gsm-n", str(a.gsm_n)]
        if cell == "residue":
            return [py, "bench/residue_sweep.py", f"t4-{arm}"]
        if cell == "soak":
            return [py, "bench/labd_soak.py", "--minutes", str(a.soak_minutes)]
        return [py, "bench/labd_bench.py", f"t4-{arm}-{cell[1:]}k", "--ctx", str(ctx[cell]), "--tasks", "qa,summary",
                "--corpus", CORPUS_LONG]

    base = {"SPEC": "dflash2", "CTX": "long", "PREFIX_CACHE": "1", "HOST": "127.0.0.1", "PORT": port}
    base.update(kv.split("=", 1) for kv in a.env)
    groups = [("residue", {}, ["residue"]),
              ("quality", {"KV_MEM": os.environ.get("T4_QUALITY_KV_MEM", "3758096384"),
                           "MAX_LEN": os.environ.get("T4_QUALITY_MAX_LEN", "65536")}, ["quality"]),
              ("deep", {"KV_MEM": os.environ.get("T4_149K_KV_MEM", "6100000000"),
                        "MAX_LEN": os.environ.get("T4_149K_MAX_LEN", "150000")}, ["d149"]),
              ("speed", {}, ["d60", "d90", "soak"])]
    boots = []
    for arm in a.arms:
        want = [c for c in a.cells if c != "soak" or arm == "fp8fa2"]
        groups_arm = [("attached", {}, [c for c in CELLS if c in want])] if a.attach else groups
        for name, extra, cells in groups_arm:
            cells = [c for c in cells if c in want]
            if cells:
                timeout = [a.cell_timeout if c != "soak" else a.soak_minutes * 60 + 3600 for c in cells]
                boots.append({"arm": arm, "name": name, "env": {**base, **extra, "FA2_FP8KV": ARMS[arm]},
                              "cells": [{"name": c, "argv": argv(arm, c), "timeout": t}
                                        for c, t in zip(cells, timeout)]})
    return boots


# ---- parsers: the scripts' own output lines ---------------------------------------------------------------

def parse_labd(text):
    rows = [dict(task=m[1], prompt=int(m[2]), tok_step=float(m[3]), decode=float(m[4]), ttft=float(m[5]))
            for m in re.finditer(r"^LABD \S+ (qa|summary|copy|code|edit|quote)\s+prompt=\s*(\d+) .*?tok/step=\s*([\d.]+) "
                                 r"decode=\s*([\d.]+) .*?ttft=\s*([\d.]+)s", text, re.M)]
    t = re.search(r"^LABD \S+ TOTAL .*?tok/step=([\d.]+) decode=([\d.]+) tok/s", text, re.M)
    if not rows or not t:
        return {}
    return {"prompt_tokens": rows[0]["prompt"], "decode": float(t[2]), "tok_step": float(t[1]),
            "ttft_cold": rows[0]["ttft"], "rows": rows}


def parse_quality(text):
    p = re.search(r" PPL (\{.*\})", text)
    g = re.search(r"GSM8K n=(\d+) acc=([\d.]+)", text)
    if not (p and g):
        return {}
    ppl = {k: v[0] for k, v in ast.literal_eval(p[1]).items()}
    return {"gsm8k": float(g[2]), "gsm_n": int(g[1]), "ppl": ppl, "ppl_all": ppl.get("all")}


def parse_residue(text):
    m = re.search(r"DONE: (\d+) broken of (\d+) residues", text)
    return {"broken": int(m[1]), "of": int(m[2])} if m else {}


def parse_soak(text):
    m = re.search(r"^soak: (\d+) rounds x (\d+) requests, tokens/step=([\d.]+), (OK|\d+ PROBLEMS)", text, re.M)
    return {"rounds": int(m[1]), "conc": int(m[2]), "tok_step": float(m[3]), "ok": m[4] == "OK"} if m else {}


def parse_boot_log(text):
    """The pool, the attention-backend lines, and whether the FP8 FA2 path announced itself."""
    pools = [int(m.replace(",", "")) for m in re.findall(r"GPU KV cache size:\s*([\d,]+) tokens", text)]
    lines = [ln.strip() for ln in text.splitlines()]
    kv = [ln for ln in lines if "GPU KV cache size" in ln]
    be = [ln for ln in lines if re.search(r"(?i)attention backend|attn.*backend|backend.*attn|FP8 FA2", ln)]
    return {"pool_tokens": pools[0] if pools else None, "kv_lines": kv[:4], "backend_lines": list(dict.fromkeys(be))[:12],
            "fa2_fp8_line": any("FP8 FA2" in ln for ln in lines)}


PARSERS = {"quality": parse_quality, "residue": parse_residue, "soak": parse_soak,
           "d60": parse_labd, "d90": parse_labd, "d149": parse_labd}


# ---- arm identity: does PRINT_ARGV show the arm we think we are booting ----------------------------------

def check_argv(arm, lines):
    """Reasons the launcher's argv is not this arm's. A launcher without the FA2_FP8KV knob ignores it, and the
    fp8fa2 arm would silently measure int8 twice."""
    def val(flag):
        return lines[lines.index(flag) + 1] if flag in lines and lines.index(flag) + 1 < len(lines) else None
    backend, dtype = val("--attention-backend"), val("--kv-cache-dtype")
    want = ("TRITON_ATTN", "int8_per_token_head") if arm == "int8" else ("FLASH_ATTN", "fp8")
    out = [f"{f} is {g!r}, the {arm} arm needs {w!r}" for f, g, w in
           (("--attention-backend", backend, want[0]), ("--kv-cache-dtype", dtype, want[1])) if g != w]
    if arm == "fp8fa2":
        try:
            spec = json.loads(val("--speculative-config") or "")
        except ValueError:
            spec = {}
        if spec.get("attention_backend") != "FLASH_ATTN":
            out.append('--speculative-config lacks "attention_backend":"FLASH_ATTN"')
    return out


# ---- the restart gate ------------------------------------------------------------------------------------

def proc_snapshot(root, proc="/proc"):
    """{server: [pid, start] | None, engines: [[pid, start]...]}: root and its EngineCore descendants, read from
    /proc (start is stat field 22, so a recycled PID is a different process)."""
    info, kids = {}, {}
    for d in os.listdir(proc):
        if not d.isdigit():
            continue
        try:
            stat = open(f"{proc}/{d}/stat").read()
            cmd = open(f"{proc}/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        f = stat[stat.rindex(")") + 2:].split()  # fields after "(comm) ": state ppid ... starttime is f[19]
        if f[0] in "ZX":  # killed and not yet reaped (our own Popen child, until poll()): dead
            continue
        info[int(d)] = (int(f[19]), stat[stat.index("(") + 1:stat.rindex(")")] + " " + cmd)
        kids.setdefault(int(f[1]), []).append(int(d))
    engines, todo = [], [root]
    while todo:
        for k in kids.get(todo.pop(), []):
            todo.append(k)
            if re.search(r"EngineCor", info[k][1]):
                engines.append([k, info[k][0]])
    return {"server": [root, info[root][0]] if root in info else None, "engines": sorted(engines)}


def docker_snapshot(name):
    r = subprocess.run(["docker", "inspect", "-f", "{{.RestartCount}}|{{.State.StartedAt}}|{{.State.Running}}", name],
                       capture_output=True, text=True)
    c, started, running = (r.stdout.strip().split("|") + ["", "", ""])[:3]
    return {"container": name, "restart_count": int(c) if c.isdigit() else None, "started_at": started,
            "running": running == "true"}


def restart_gate(base, after, log_new):
    """Reasons the cell fails the restart gate: any change in the server's identity since the boot's baseline,
    or an EngineCore death line in the log the cell produced."""
    why = []
    if "container" in base:
        if after.get("restart_count") != base.get("restart_count") or after.get("started_at") != base.get("started_at"):
            why.append(f"container restarted: RestartCount {base.get('restart_count')} -> {after.get('restart_count')}")
        if not after.get("running"):
            why.append("container not running")
    else:
        if after["server"] != base["server"]:
            why.append(f"server process changed: {base['server']} -> {after['server']}")
        if after["engines"] != base["engines"]:
            why.append(f"EngineCore processes changed: {base['engines']} -> {after['engines']}")
    why += [f"EngineCore death in the log: {ln.strip()[:200]}" for ln in log_new.splitlines() if DEATH.search(ln)]
    return why


def kernel_xids():
    """Xid lines from the kernel log, or None when neither dmesg nor journalctl -k is readable."""
    for cmd in (["dmesg"], ["journalctl", "-k", "--no-pager"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            return [ln for ln in r.stdout.splitlines() if "Xid" in ln]
    return None


class Target:
    """What the gate watches: a launched or attached server's identity and its log."""

    def __init__(self, pid=None, container=None, log=None):
        self.pid, self.container, self.log = pid, container, log

    def snapshot(self):
        return docker_snapshot(self.container) if self.container else proc_snapshot(self.pid)

    def log_text(self):
        if self.container and not self.log:
            return subprocess.run(["docker", "logs", self.container], capture_output=True, text=True).stdout
        return open(self.log, errors="replace").read() if self.log and os.path.exists(self.log) else ""


# ---- judging and the verdict ------------------------------------------------------------------------------

def judge(cell, rc, m):
    """(status, reasons) from the exit code and parsed metrics, before the restart gate. 'info' is reported only."""
    if cell == "residue" and m and m["broken"]:
        return "fail", [f"{m['broken']} of {m['of']} residues broken" + (f", exit {rc}" if rc not in (0, 1) else "")]
    if rc != 0 or not m:
        return ("fail" if cell in ("quality", "residue", "soak", "d90") else "info"), \
            [f"exit {rc}" + ("" if m else ", output not parsed")]
    if cell == "soak" and not m["ok"]:
        return "fail", ["soak reported problems"]
    return ("info" if cell in ("d60", "d90", "d149") else "pass"), []


def pct(a, b):
    return None if not a or b is None else (b / a - 1) * 100


def evaluate(res, arms, cells):
    """The bars, as [{name, status: pass|fail|not-evaluated|n/a, detail}], and the verdict."""
    g = []

    def cell(arm, c):
        return res.get(arm, {}).get("cells", {}).get(c)

    for arm in arms:
        bad = [f"{c['name']}: {'; '.join(c['gate_reasons'])}" for c in res.get(arm, {}).get("cells", {}).values()
               if c.get("gate_reasons")]
        ran = bool(res.get(arm, {}).get("cells"))
        g.append({"name": f"{arm}: restart gate", "status": "fail" if bad else "pass" if ran else "not-evaluated",
                  "detail": "; ".join(bad) or ("" if ran else "no cell ran")})
        for c in ("quality", "residue", "soak"):
            if c in cells and (c != "soak" or arm == "fp8fa2"):
                r = cell(arm, c)
                g.append({"name": f"{arm}: {c}", "status": r["status"] if r and r["status"] in ("pass", "fail")
                          else "not-evaluated", "detail": "; ".join(r["reasons"]) if r else "cell did not run"})
    cross = len(arms) == 2
    a, b = cell("int8", "quality"), cell("fp8fa2", "quality")
    if "quality" in cells:
        if not cross:
            g.append({"name": "fp8fa2 vs int8: quality", "status": "n/a", "detail": "one arm"})
        elif a and b and not (a.get("metrics") and b.get("metrics")):
            g.append({"name": "fp8fa2 vs int8: quality", "status": "fail", "detail": "a battery did not complete"})
        elif a and b:
            drop = (a["metrics"]["gsm8k"] - b["metrics"]["gsm8k"]) * 100
            rise = pct(a["metrics"]["ppl_all"], b["metrics"]["ppl_all"])
            ok = drop <= BARS["gsm_drop_pt"] and rise <= BARS["ppl_rise_pct"]
            g.append({"name": "fp8fa2 vs int8: quality", "status": "pass" if ok else "fail",
                      "detail": f"GSM8K {-drop:+.1f} pt (bar -{BARS['gsm_drop_pt']}), PPL all {rise:+.3f}% "
                                f"(bar +{BARS['ppl_rise_pct']}%)"})
        else:
            g.append({"name": "fp8fa2 vs int8: quality", "status": "not-evaluated", "detail": "no battery numbers"})
    if "d90" in cells:
        a, b = cell("int8", "d90"), cell("fp8fa2", "d90")
        if not cross:
            g.append({"name": "fp8fa2 vs int8: decode at ~90k", "status": "n/a", "detail": "one arm"})
        elif a and b and not (a.get("metrics") and b.get("metrics")):
            g.append({"name": "fp8fa2 vs int8: decode at ~90k", "status": "fail", "detail": "a d90 cell did not complete"})
        elif a and b:
            x, y = a["metrics"]["decode"], b["metrics"]["decode"]
            g.append({"name": "fp8fa2 vs int8: decode at ~90k", "status": "pass" if y >= x else "fail",
                      "detail": f"{y} vs {x} tok/s ({pct(x, y):+.1f}%)"})
        else:
            g.append({"name": "fp8fa2 vs int8: decode at ~90k", "status": "not-evaluated", "detail": "no decode numbers"})
    st = {x["status"] for x in g}
    return g, ("FAIL" if "fail" in st else "INCOMPLETE" if "not-evaluated" in st else
               "PARTIAL" if "n/a" in st else "PASS")


def fmt(v, spec=""):
    return "-" if v is None else format(v, spec)


def render_md(s):
    res = s["arms"]
    arms = list(res)

    def m(arm, c, key, spec=""):
        r = res.get(arm, {}).get("cells", {}).get(c)
        return fmt((r or {}).get("metrics", {}).get(key) if r and r.get("metrics") else None, spec)

    def row(label, c, key, spec="", delta=True):
        v = [res.get(a, {}).get("cells", {}).get(c) for a in ("int8", "fp8fa2")]
        x, y = [(r or {}).get("metrics", {}).get(key) if r else None for r in v]
        d = f"{pct(x, y):+.1f}%" if delta and x and y is not None else ""
        return f"| {label} | {fmt(x, spec)} | {fmt(y, spec)} | {d} |"

    def pool(arm):
        bs = res.get(arm, {}).get("boots", {})
        return (bs.get("speed") or next(iter(bs.values()), {})).get("pool_tokens")

    pi, pf = pool("int8"), pool("fp8fa2")
    out = [f"# T4 battery: {s['verdict']}", "",
           f"git {s['meta']['git_sha']}{' (dirty)' if s['meta']['git_dirty'] else ''}, host {s['meta']['host']}, "
           f"GPU {s['meta']['gpu']}, vLLM pin {s['meta']['vllm_pin']}, fa2 pin {s['meta']['fa2_pin'] or 'absent'}", "",
           "## Bars", "", "| bar | status | detail |", "|---|---|---|"]
    out += [f"| {x['name']} | {x['status']} | {x['detail']} |" for x in s["gates"]]
    out += ["", "## Numbers", "", "| | int8 + Triton | fp8 + FA2 | fp8fa2 / int8 |", "|---|---|---|---|",
            f"| pool (tokens) | {fmt(pi)} | {fmt(pf)} | {f'{pct(pi, pf):+.1f}%' if pi and pf else ''} |"]
    for c, n in (("d60", "~60k"), ("d90", "~90k"), ("d149", "~149k")):
        if any(c in res.get(a, {}).get("cells", {}) for a in arms):
            out += [row(f"{n} prompt tokens", c, "prompt_tokens", delta=False),
                    row(f"{n} decode tok/s", c, "decode", ".1f"), row(f"{n} tok/step", c, "tok_step", ".2f", False),
                    row(f"{n} cold TTFT s", c, "ttft_cold", ".1f")]
    if any("quality" in res.get(a, {}).get("cells", {}) for a in arms):
        out += [row("GSM8K n=200", "quality", "gsm8k", ".3f", False), row("PPL all", "quality", "ppl_all", ".4f")]
    out += ["| residue broken / 128 | " + " | ".join(m(a, "residue", "broken") for a in ("int8", "fp8fa2")) + " | |"]
    out += ["| soak | | " + (m("fp8fa2", "soak", "rounds") + " rounds, tok/step " + m("fp8fa2", "soak", "tok_step")) + " | |",
            "", "## Cells", "", "| arm | cell | status | rc | s | reasons | Xid new |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        for c in res.get(a, {}).get("cells", {}).values():
            x = c.get("xid_new")
            out.append(f"| {a} | {c['name']} | {c['status']} | {c.get('rc')} | {c.get('secs', '')} | "
                       f"{'; '.join(c['reasons'] + c.get('gate_reasons', []))} | "
                       f"{'unreadable' if x is None else len(x)} |")
    return "\n".join(out) + "\n"


# ---- running ----------------------------------------------------------------------------------------------

def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def meta(a):
    pin = {}
    if os.path.exists(f"{REPO}/fa2/PIN"):
        pin = dict(ln.strip().split("=", 1) for ln in open(f"{REPO}/fa2/PIN") if re.match(r"^\w+=", ln.strip()))
    req = re.search(r"^vllm==(\S+)", open(f"{REPO}/docker/requirements.txt").read(), re.M)
    gpu = sh("nvidia-smi", "--query-gpu=name,driver_version,power.limit,memory.total", "--format=csv,noheader")
    return {"started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "git_sha": sh("git", "rev-parse", "HEAD") or "unknown", "git_dirty": bool(sh("git", "status", "--porcelain")),
            "host": "WSL2" if "microsoft" in open("/proc/version").read().lower() else "native",
            "gpu": gpu or "unreadable", "vllm_pin": req[1] if req else None, "fa2_pin": pin,
            "argv": sys.argv, "arms": a.arms, "cells": a.cells, "bars": BARS}


def healthy():
    try:
        with urllib.request.urlopen(harness.base_url() + "/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def teardown(p):
    if p.poll() is None:
        for sig, wait in ((signal.SIGTERM, 180), (signal.SIGKILL, 60)):
            try:
                os.killpg(p.pid, sig)
                p.wait(wait)
                break
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
    for _ in range(60):  # the next boot needs the card back
        used = sh("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
        if not used.isdigit() or int(used) < 2000:
            break
        time.sleep(2)


def run_cmd(argv, env, path, timeout):
    with open(path, "w") as f:
        p = subprocess.Popen(argv, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = 124
    return rc, open(path, errors="replace").read()


def run_boot(a, boot, res, out):
    arm, name, env = boot["arm"], boot["name"], boot["env"]
    d = f"{out}/{arm}"
    os.makedirs(d, exist_ok=True)
    arm_res = res.setdefault(arm, {"boots": {}, "cells": {}})
    penv = {**os.environ, **env}
    info = {"env": {k: v for k, v in sorted(penv.items()) if KNOB.match(k) and not SECRET.search(k)},
            "env_added": env}
    arm_res["boots"][name] = info
    proc = None
    target = Target(a.server_pid, a.container, a.server_log)
    skip = None
    if not a.attach:
        r = subprocess.run(["bash", "single-user/start_qwen.sh"], cwd=REPO, env={**penv, "PRINT_ARGV": "1"},
                           capture_output=True, text=True)
        open(f"{d}/{name}.argv.txt", "w").write(r.stdout)
        open(f"{d}/{name}.argv.stderr.txt", "w").write(r.stderr)
        info["argv"] = r.stdout.splitlines()
        skip = (["PRINT_ARGV: launcher exit %d: %s" % (r.returncode, r.stderr.strip()[-300:])] if r.returncode
                else check_argv(arm, r.stdout.splitlines()))
        if not skip:
            if healthy():
                skip = ["a server already answers on this port; stop it, or use --attach"]
            else:
                log = f"{d}/{name}.server.log"
                proc = subprocess.Popen(["bash", "single-user/start_qwen.sh"], cwd=REPO, env=penv, start_new_session=True,
                                        stdout=open(log, "w"), stderr=subprocess.STDOUT)
                target = Target(proc.pid, None, log)
                t0 = time.time()
                while not healthy():
                    if proc.poll() is not None or time.time() - t0 > a.boot_timeout:
                        skip = ["boot failed: " + ("server exited %s" % proc.poll() if proc.poll() is not None
                                                   else "no /health in %ds" % a.boot_timeout)]
                        break
                    time.sleep(2)
                info["boot_s"] = round(time.time() - t0)
                info["log"] = log
    try:
        if not skip:
            info.update(parse_boot_log(target.log_text()))
            info["gpu_mem_after_boot"] = sh("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader")
            base = target.snapshot()
            info["baseline"] = base
        for c in boot["cells"]:
            r = {"name": c["name"], "cmd": c["argv"], "reasons": [], "gate_reasons": []}
            arm_res["cells"][c["name"]] = r
            if skip:
                r.update(status="fail" if c["name"] in ("quality", "residue", "soak", "d90") else "info",
                         reasons=["not run: " + "; ".join(skip)])
                continue
            print(f"[{arm}/{name}] {c['name']}: {' '.join(c['argv'])}", flush=True)
            before, x0, mark = target.snapshot(), kernel_xids(), len(target.log_text())
            t0 = time.time()
            rc, text = run_cmd(c["argv"], os.environ, f"{d}/{c['name']}.log", c["timeout"])
            after, x1 = target.snapshot(), kernel_xids()
            m = PARSERS[c["name"]](text)
            r.update(rc=rc, secs=round(time.time() - t0), metrics=m, before=before, after=after,
                     xid_new=None if x0 is None or x1 is None else [ln for ln in x1 if ln not in x0])
            r["status"], r["reasons"] = judge(c["name"], rc, m)
            r["gate_reasons"] = restart_gate(base, after, target.log_text()[mark:])
            if r["gate_reasons"]:
                r["status"] = "fail"
            print(f"[{arm}/{name}] {c['name']}: {r['status']} {r['reasons'] + r['gate_reasons']}", flush=True)
            write_summary(a, res, out)
            if r["gate_reasons"]:
                skip = ["server restarted or died during " + c["name"]]
    finally:
        if proc:
            teardown(proc)
    write_summary(a, res, out)


def write_summary(a, res, out):
    gates, verdict = evaluate(res, a.arms, a.cells)
    s = {"verdict": verdict, "meta": a._meta, "gates": gates, "arms": res}
    json.dump(s, open(f"{out}/summary.json", "w"), indent=1)
    open(f"{out}/summary.md", "w").write(render_md(s))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="int8,fp8fa2", help="comma list of int8, fp8fa2 (default both)")
    ap.add_argument("--cells", default=",".join(DEFAULT_CELLS), help=f"comma list of {', '.join(CELLS)} "
                    f"(default {','.join(DEFAULT_CELLS)}; d149 is opt-in; soak runs on fp8fa2 only)")
    ap.add_argument("--soak-minutes", type=int, default=240)
    ap.add_argument("--gsm-n", type=int, default=200)
    ap.add_argument("--ctx60", type=int, default=53000)
    ap.add_argument("--ctx90", type=int, default=85000)
    ap.add_argument("--ctx149", type=int, default=153000)
    ap.add_argument("--out", help="results directory (default bench/results/t4-battery-<UTC>)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL", help="added to every boot")
    ap.add_argument("--dry-run", action="store_true", help="print every command and env, start nothing")
    ap.add_argument("--attach", choices=list(ARMS), help="a server for this one arm is already up: boot nothing, "
                    "run the cells against it (the residue cell is then warm, not a cold restart)")
    ap.add_argument("--server-pid", type=int, help="with --attach: the server's PID (native)")
    ap.add_argument("--container", help="with --attach: the docker container, gated on its RestartCount")
    ap.add_argument("--server-log", help="with --attach: the server's log file, scanned for EngineCore deaths")
    ap.add_argument("--boot-timeout", type=int, default=1800, help="seconds to wait for /health (default 1800)")
    ap.add_argument("--cell-timeout", type=int, default=4 * 3600, help="seconds per cell except soak (default 14400)")
    a = ap.parse_args()
    a.arms = a.arms.split(",")
    a.cells = a.cells.split(",")
    if a.attach:
        if a.arms == ["int8", "fp8fa2"] and "--arms" not in sys.argv:
            a.arms = [a.attach]
        if a.arms != [a.attach]:
            ap.error("--attach ARM runs one arm: --arms must be that arm")
        if not (a.server_pid or a.container):
            ap.error("--attach needs --server-pid or --container, to watch the server's identity")
    for x, ok in ((a.arms, ARMS), (a.cells, CELLS)):
        if any(i not in ok for i in x):
            ap.error(f"unknown name in {x}; known: {list(ok)}")
    boots = plan(a)
    out = os.path.abspath(a.out or f"{REPO}/bench/results/t4-battery-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    if a.dry_run:
        print(f"# t4_battery dry run: arms={a.arms} cells={a.cells} out={out}\n# nothing is started.")
        for b in boots:
            print(f"\n== {b['arm']} / boot {b['name']}")
            if not a.attach:
                env = " ".join(f"{k}={v}" for k, v in b["env"].items())
                print(f"  PRINT_ARGV=1 {env} bash single-user/start_qwen.sh   # must show the {b['arm']} arm's backend")
                print(f"  {env} bash single-user/start_qwen.sh > {out}/{b['arm']}/{b['name']}.server.log")
            for c in b["cells"]:
                print(f"  cell {c['name']}: {' '.join(c['argv'])}   # timeout {c['timeout']}s, restart gate before/after")
        return 0
    missing = [p for p in (os.environ.get("PY", f"{REPO}/venv/bin/python"), os.path.expanduser("~/bench/labd_corpus.txt"),
                           os.path.expanduser(CORPUS_LONG), f"{HERE}/quality-data") if not os.path.exists(p)]
    if missing:
        sys.exit("missing: " + ", ".join(missing) + "\n(corpora: bench/labd_bench.py once, then bench/make_long_corpus.py; "
                 "quality data: the hf download lines in bench/quality_battery.py)")
    os.makedirs(out, exist_ok=True)
    a._meta = meta(a)
    res = {}
    try:
        for b in boots:
            run_boot(a, b, res, out)
    finally:
        s = write_summary(a, res, out)
    print(f"\n{s['verdict']}: {out}/summary.md")
    return {"PASS": 0, "PARTIAL": 0, "FAIL": 1}.get(s["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
