"""Builds Project Maya's `/metrics` JSON (what its dashboard draws) from a vLLM server.

Sources: vLLM's Prometheus endpoint, /v1/models, nvidia-smi, /proc, and the
"[hyperqwen] ..." request-progress lines in the server log (live phase, per-request rows).
Standard library only.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import urllib.request
from collections import deque

SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([-+.\deE]+|NaN|[+-]?Inf)(?:\s+\d+)?$')
REASON = re.compile(r'finished_reason="([^"]+)"')
NUM = r"([\d,]+(?:\.\d+)?)"
RE_READ = re.compile(rf"reading the prompt: {NUM} of {NUM} tokens, {NUM} s so far \(req ([\w-]+)\)")
RE_PREFILL = re.compile(rf"prefill: {NUM} new tokens in {NUM} s \({NUM} tok/s\), cached {NUM} of {NUM} \(req ([\w-]+)\)")
RE_ANSWER = re.compile(rf"answering: {NUM} of max {NUM} tokens, {NUM} tok/s, {NUM} s \(req ([\w-]+)\)")
RE_DONE = re.compile(rf"done: {NUM} tokens in {NUM} s \({NUM} tok/s\) \((\w+), cancel=(\w+)\) \(req ([\w-]+)\)")
HIST = 60


def _f(s: str) -> float:
    return float(s.replace(",", ""))


class Source:
    def __init__(self, url: str, key: str, log_path: str):
        self.url, self.key, self.log_path = url.rstrip("/"), key, log_path
        self.hist: dict[str, deque] = {k: deque(maxlen=HIST) for k in (
            "tok_s", "gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "cpu", "ram_used", "disk_read_mb",
            "gpu_pcie_rx_mb")}
        self.max_context = None
        self.prev: tuple[float, float] | None = None       # (generation tokens, time)
        self.cpu_prev = None
        self.disk_prev = None
        self.reqs: deque = deque(maxlen=30)
        self.cur: dict = {}                                 # request id -> its live facts
        self.log_pos = None
        self.pending = ""
        self.first_log = True

    # ---------------------------------------------------------------- server
    def _get(self, path: str) -> str:
        req = urllib.request.Request(self.url + path,
                                     headers={"Authorization": "Bearer " + self.key} if self.key else {})
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.read().decode(errors="replace")

    def prom(self) -> tuple[dict[str, float], dict[str, float]]:
        out: dict[str, float] = {}
        success: dict[str, float] = {}
        for line in self._get("/metrics").splitlines():
            m = SAMPLE.match(line)
            if not m:
                continue
            name, labels, raw = m.groups()
            try:
                v = float(raw)
            except ValueError:
                continue
            if not math.isfinite(v):
                continue
            out[name] = out.get(name, 0.0) + v
            if name == "vllm:request_success_total" and labels:
                r = REASON.search(labels)
                if r:
                    success[r.group(1)] = success.get(r.group(1), 0.0) + v
        return out, success

    # ----------------------------------------------------------------- log
    def read_log(self) -> None:
        try:
            size = os.path.getsize(self.log_path)
            with open(self.log_path, "rb") as f:
                if self.log_pos is None:
                    f.seek(max(0, size - 200_000))
                    stamp = os.path.getmtime(self.log_path)
                elif size < self.log_pos:
                    f.seek(0)
                    stamp = time.time()
                else:
                    f.seek(self.log_pos)
                    stamp = time.time()
                data = f.read()
                self.log_pos = f.tell()
        except OSError:
            return
        text = self.pending + data.decode(errors="replace")
        lines = text.split("\n")
        self.pending = lines.pop()
        for line in lines:
            self.parse(line, stamp)
        self.first_log = False

    def parse(self, line: str, stamp: float) -> None:
        if "[hyperqwen]" not in line:
            return
        line = line[line.index("[hyperqwen]"):]
        now = stamp if self.first_log else time.time()
        if m := RE_READ.search(line):
            read, total, secs, rid = m.groups()
            c = self.cur.setdefault(rid, {"t0": now, "id": rid})
            c.update(state="reading", prompt_read=_f(read), prompt_total=_f(total), updated=now)
            c.setdefault("read0", _f(read))
        elif m := RE_PREFILL.search(line):
            new, secs, rate, cached, total, rid = m.groups()
            c = self.cur.setdefault(rid, {"t0": now, "id": rid})
            c.update(prefill_tok_s=_f(rate), prompt_tokens=_f(total), reused=_f(cached), prompt_ms=_f(secs) * 1000,
                     state="answering", generated=0.0, updated=now)
        elif m := RE_ANSWER.search(line):
            gen, mx, rate, secs, rid = m.groups()
            c = self.cur.setdefault(rid, {"t0": now, "id": rid})
            c.update(state="answering", generated=_f(gen), max_tokens=_f(mx), tok_s=_f(rate), elapsed_s=_f(secs),
                     updated=now)
        elif m := RE_DONE.search(line):
            toks, secs, rate, reason, cancel, rid = m.groups()
            c = self.cur.pop(rid, {})
            total = c.get("prompt_tokens") or c.get("prompt_total") or 0
            reused = c.get("reused") or 0
            self.reqs.appendleft({
                "time": now, "finish": "cancel" if cancel == "True" else reason, "api": "openai",
                "prompt_tokens": total, "reused": reused, "prefill_tok_s": c.get("prefill_tok_s"),
                "prompt_ms": c.get("prompt_ms"), "output_tokens": _f(toks), "decode_tok_s": _f(rate),
                "duration_s": _f(secs), "hit_rate": reused / total if total else None})

    def live(self, running: float, waiting: float) -> dict:
        now = time.time()
        for rid in [k for k, c in self.cur.items() if now - c.get("updated", 0) > 120]:
            self.cur.pop(rid)                               # (a request that never logged its end)
        if not running and not waiting:
            return {"state": "idle"}
        if not self.cur:
            return {"state": "generating" if running else "idle", "queued": int(waiting)}
        c = max(self.cur.values(), key=lambda c: c.get("updated", 0))
        out = {"queued": int(waiting)}
        if c["state"] == "reading":
            dt = now - c["t0"]
            rate = (c["prompt_read"] - c["read0"]) / dt if dt > 1 and c["prompt_read"] > c["read0"] else None
            out.update(state="reading", prompt_read=c["prompt_read"], prompt_total=c["prompt_total"],
                       prompt_tokens=c["prompt_total"], prefill_tok_s_mean=rate)
        else:
            out.update(state="generating", tok_s=c.get("tok_s"), generated=c.get("generated") or 0,
                       elapsed_s=c.get("elapsed_s") or (now - c["t0"]), max_tokens=c.get("max_tokens"),
                       prompt_tokens=c.get("prompt_tokens") or 0)
        return out

    # ------------------------------------------------------------ hardware
    def gpus(self) -> list[dict]:
        cmd = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,"
               "power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current",
               "--format=csv,noheader,nounits"]
        try:
            data = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2)
        except (OSError, subprocess.SubprocessError):
            return []

        def n(s, scale=1.0):
            try:
                return float(s) * scale
            except ValueError:
                return None

        def i(s):
            v = n(s)
            return None if v is None else int(v)
        cards = []
        for line in data.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) != 11:
                continue
            cards.append({"index": int(p[0]), "name": p[1], "util": n(p[2]), "mem_used": n(p[3], 2**20),
                          "mem_total": n(p[4], 2**20), "temp": n(p[5]), "power": n(p[6]), "power_limit": n(p[7]),
                          "pcie_gen": i(p[8]), "pcie_gen_max": i(p[9]), "pcie_width": i(p[10])})
        return cards

    def cpu(self):
        try:
            with open("/proc/stat") as f:
                v = [int(x) for x in f.readline().split()[1:]]
        except (OSError, ValueError):
            return None
        idle, total = v[3] + v[4], sum(v)
        prev, self.cpu_prev = self.cpu_prev, (idle, total)
        if not prev or total == prev[1]:
            return None
        return 100.0 * (1 - (idle - prev[0]) / (total - prev[1]))

    @staticmethod
    def ram():
        try:
            with open("/proc/meminfo") as f:
                info = {k: int(v.split()[0]) * 1024 for k, v in (l.split(":") for l in f)}
            return info["MemTotal"] - info["MemAvailable"], info["MemTotal"]
        except (OSError, KeyError, ValueError):
            return None, None

    def disk(self):
        now, read, write = time.monotonic(), 0, 0
        try:
            with open("/proc/diskstats") as f:
                for l in f:
                    p = l.split()
                    if re.fullmatch(r"nvme\d+n\d+|sd[a-z]+|vd[a-z]+", p[2]):
                        read += int(p[5]) * 512
                        write += int(p[9]) * 512
        except (OSError, IndexError, ValueError):
            return None, None
        prev, self.disk_prev = self.disk_prev, (read, write, now)
        if not prev or now == prev[2]:
            return None, None
        dt = now - prev[2]
        return (read - prev[0]) / dt / 1e6, (write - prev[1]) / dt / 1e6

    # --------------------------------------------------------------- build
    def snapshot(self) -> dict:
        d, success = self.prom()
        self.read_log()
        if self.max_context is None:
            try:
                self.max_context = json.loads(self._get("/v1/models"))["data"][0].get("max_model_len") or 0
            except Exception:
                self.max_context = 0
        now = time.monotonic()
        gen = d.get("vllm:generation_tokens_total", d.get("vllm:generation_tokens", 0.0))
        rate = 0.0
        if self.prev:
            rate = max(0.0, gen - self.prev[0]) / max(0.001, now - self.prev[1])
        self.prev = (gen, now)
        running, waiting = d.get("vllm:num_requests_running", 0.0), d.get("vllm:num_requests_waiting", 0.0)
        live = self.live(running, waiting)
        if live["state"] == "generating" and live.get("tok_s") is None:
            live["tok_s"] = rate

        cards = self.gpus()
        cpu, (ram_used, ram_total) = self.cpu(), self.ram()
        disk_r, disk_w = self.disk()
        hw: dict = {"gpus": cards, "cpu": cpu, "ram_used": ram_used, "ram_total": ram_total,
                    "disk_read_mb": disk_r, "disk_write_mb": disk_w}
        if cards:
            def vals(k):
                return [c[k] for c in cards if c.get(k) is not None]
            hw.update(gpu_util=sum(vals("util")) / len(cards), gpu_mem_used=sum(vals("mem_used")),
                      gpu_mem_total=sum(vals("mem_total")), gpu_temp=max(vals("temp") or [0]),
                      gpu_power=sum(vals("power")), gpu_power_limit=sum(vals("power_limit")),
                      gpu_pcie_gen=cards[0]["pcie_gen"], gpu_pcie_gen_max=cards[0]["pcie_gen_max"],
                      gpu_pcie_width=cards[0]["pcie_width"])
        for key, val in (("tok_s", rate), ("gpu_util", hw.get("gpu_util")), ("gpu_mem_used", hw.get("gpu_mem_used")),
                         ("gpu_temp", hw.get("gpu_temp")), ("gpu_power", hw.get("gpu_power")), ("cpu", cpu),
                         ("ram_used", ram_used), ("disk_read_mb", disk_r)):
            self.hist[key].append(val or 0.0)

        cores = None
        try:
            with open("/proc/cpuinfo") as f:
                m = re.search(r"cpu cores\s*:\s*(\d+)", f.read())
                cores = int(m.group(1)) if m else None
        except OSError:
            pass
        totals = {"requests": sum(success.values()), "prompt_tokens": d.get("vllm:prompt_tokens_total", 0.0),
                  "reused": d.get("vllm:prompt_tokens_cached_total", 0.0), "output_tokens": gen,
                  "prompt_ms": d.get("vllm:request_prefill_time_seconds_sum", 0.0) * 1000,
                  "decode_ms": d.get("vllm:request_decode_time_seconds_sum", 0.0) * 1000,
                  "since": d.get("process_start_time_seconds") or time.time()}
        return {"live": live, "history": {k: list(v) for k, v in self.hist.items()},
                "engine": {"max_context": self.max_context, "engine_kind": "vllm"},
                "totals": totals, "requests": list(self.reqs), "tiers": {"now": None},
                "hardware": hw,
                "hardware_static": {"gpu_name": " + ".join(c["name"] for c in cards), "cores": cores,
                                    "threads": os.cpu_count()}}
