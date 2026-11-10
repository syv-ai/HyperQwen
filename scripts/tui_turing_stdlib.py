#!/usr/bin/env python3
"""Standard-library fallback display for tui_turing.py (no Textual/Rich needed)."""
from __future__ import annotations

import curses
import math
import os
import re
import subprocess
import time
import urllib.request

URL = os.environ.get("HYPERQWEN_URL", "http://127.0.0.1:18020")
LOG = os.environ.get("HYPERQWEN_LOG_FILE", "logs/turing.log")
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([-+.\deE]+)(?:\s+\d+)?$")
BLOCKS = "▁▂▃▄▅▆▇█"


def scrape() -> dict[str, float]:
    headers = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]} if os.environ.get("VLLM_API_KEY") else {}
    req = urllib.request.Request(URL.rstrip("/") + "/metrics", headers=headers)
    with urllib.request.urlopen(req, timeout=3) as response:
        body = response.read().decode(errors="replace")
    out: dict[str, float] = {}
    for line in body.splitlines():
        match = SAMPLE.match(line)
        if match:
            name, raw = match.groups()
            value = float(raw)
            if math.isfinite(value):
                out[name] = out.get(name, 0.0) + value
    return out


def gpus() -> list[str]:
    cmd = ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return ["nvidia-smi unavailable"]
    rows = []
    for row in result.splitlines():
        c = [v.strip() for v in row.split(",")]
        if len(c) == 8:
            i, name, util, used, total, temp, watts, limit = c
            name = name.replace("NVIDIA ", "").replace("GeForce ", "")
            rows.append(f"GPU {i} {name}  {util}%  VRAM {used}/{total} MiB  {temp}°C  {watts}/{limit} W")
    return rows or ["No GPU readings"]


def request_lines() -> list[str]:
    try:
        with open(LOG, errors="replace") as stream:
            lines = stream.readlines()[-100:]
    except OSError:
        return [f"Waiting for {LOG}"]
    return [line[line.index("[hyperqwen]"):].rstrip() for line in lines if "[hyperqwen]" in line][-8:]


def value(data: dict[str, float], *names: str) -> float | None:
    return next((data[n] for n in names if n in data), None)


def short(n: float | None) -> str:
    return "–" if n is None else f"{n:,.1f}"


def main() -> None:
    def run(screen):
        curses.curs_set(0)
        screen.timeout(1000)
        offset = 0
        previous: tuple[float, float] | None = None
        history: list[float] = []
        while True:
            try:
                data = scrape()
                error = ""
            except Exception as exc:
                data, error = {}, str(exc)
            total = value(data, "vllm:generation_tokens_total", "vllm:generation_tokens")
            rate = 0.0
            now = time.monotonic()
            if total is not None and previous:
                rate = max(0.0, total - previous[0]) / max(0.001, now - previous[1])
            if total is not None:
                previous = (total, now)
            history = (history + [rate])[-50:]
            kv = value(data, "vllm:kv_cache_usage_perc")
            running = value(data, "vllm:num_requests_running")
            waiting = value(data, "vllm:num_requests_waiting")
            metrics = sorted(data.items())
            spark = "".join(BLOCKS[min(7, int(max(0.0, n) / max(max(history), 1e-9) * 7))] for n in history)
            lines = [
                f" HYPERQWEN / TURING     {URL}     {time.strftime('%H:%M:%S')}     q: quit | ↑↓: scroll metrics",
                "",
                f" SERVER  {'UNAVAILABLE: ' + error if error else 'ONLINE'}    {len(data)} metric series",
                f" LIVE    decode {rate:,.1f} tok/s  {spark}",
                f"         running {short(running)}  waiting {short(waiting)}  KV cache {short(kv * 100 if kv is not None else None)}%",
                "",
                " GPU HARDWARE",
                *["  " + row for row in gpus()],
                "",
                " RECENT REQUESTS",
                *["  " + row for row in request_lines()],
                "",
                f" ALL PROMETHEUS METRICS  ({len(metrics)} series)",
                *[f"  {name:<68} {number:>16,.3f}" for name, number in metrics],
            ]
            try:
                height, width = screen.getmaxyx()
                screen.erase()
                for row, line in enumerate(lines[offset:offset + height - 1]):
                    style = curses.A_BOLD if row == 0 or line.startswith((" SERVER", " GPU ", " RECENT", " ALL ")) else curses.A_NORMAL
                    screen.addnstr(row, 0, line, max(1, width - 1), style)
                screen.refresh()
            except curses.error:
                pass
            key = screen.getch()
            if key in (ord("q"), ord("Q"), 3):
                return
            if key in (curses.KEY_DOWN, ord("j")):
                offset = min(max(0, len(lines) - height + 1), offset + 1)
            elif key in (curses.KEY_UP, ord("k")):
                offset = max(0, offset - 1)
            elif key == curses.KEY_NPAGE:
                offset = min(max(0, len(lines) - height + 1), offset + height - 2)
            elif key == curses.KEY_PPAGE:
                offset = max(0, offset - height + 2)

    try:
        curses.wrapper(run)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
