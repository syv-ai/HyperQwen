#!/usr/bin/env python3
"""Follow per-request progress from the Turing server log.

The output format is modeled on Project Maya's (MIT, https://github.com/mw00/project-maya).

Default: reading the prompt, prefill tok/s, answering progress and done lines.
Use --metrics for aggregate Prometheus counters, or --once for startup totals.
No request text is displayed. The server keeps running when the watcher exits.
"""
import argparse
import datetime
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([-+.\deE]+)(?:\s+\d+)?$')
REQUEST_METRICS = (
    'request_prompt_tokens', 'request_generation_tokens',
    'request_prefill_kv_computed_tokens', 'request_prefill_time_seconds',
    'request_decode_time_seconds',
)


def parse_metrics(body):
    result = {}
    for line in body.splitlines():
        match = SAMPLE.match(line)
        if match:
            key, raw = match.groups()
            value = float(raw)
            if math.isfinite(value):
                result[key] = result.get(key, 0.0) + value
    return result


def read_metrics(url):
    headers = {}
    if os.environ.get('VLLM_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['VLLM_API_KEY']
    req = urllib.request.Request(url.rstrip('/') + '/metrics', headers=headers)
    with urllib.request.urlopen(req, timeout=5) as response:
        return parse_metrics(response.read().decode())


def stamp():
    return datetime.datetime.now().strftime('%H:%M:%S')


def completed_summary(previous, current):
    counts = [current.get('vllm:' + name + '_count') for name in REQUEST_METRICS]
    if None in counts or len(set(counts)) != 1:
        return None  # A scrape can straddle a metrics update; wait for consistency.
    count = counts[0] - previous.get('vllm:request_prompt_tokens_count', 0)
    if count <= 0:
        return None
    def delta(name):
        key = 'vllm:' + name + '_sum'
        return current[key] - previous.get(key, 0)
    prompt = delta('request_prompt_tokens')
    generated = delta('request_generation_tokens')
    computed = delta('request_prefill_kv_computed_tokens')
    prefill = delta('request_prefill_time_seconds')
    decode = delta('request_decode_time_seconds')
    def rate(tokens, seconds):
        return f'{tokens / seconds:.1f} tok/s' if seconds > 0 else 'n/a'
    return (
        f'{int(count)} completed | prompt={int(prompt)} cached={int(max(0, prompt - computed))} '
        f'new={int(computed)} | prefill {prefill:.3f}s '
        f'({rate(computed, prefill)}) | generated={int(generated)} | '
        f'decode {decode:.3f}s ({rate(max(0, generated - count), decode)})'
    )


def follow_log(path, interval, lines):
    marker = "[hyperqwen]"
    print(f"Following request progress in {path} — Ctrl+C stops watching.", flush=True)
    handle = None
    identity = None
    waiting = False
    try:
        while True:
            try:
                info = path.stat()
            except FileNotFoundError:
                if not waiting:
                    print("Waiting for the Turing server log...", flush=True)
                    waiting = True
                time.sleep(interval)
                continue
            file_id = (info.st_dev, info.st_ino)
            if handle is None or identity != file_id or info.st_size < handle.tell():
                if handle is not None:
                    handle.close()
                handle = path.open(errors="replace")
                identity = file_id
                offset = max(0, info.st_size - 65536)
                handle.seek(offset)
                if offset:
                    handle.readline()
                recent = [line[line.index(marker):].rstrip() for line in handle if marker in line]
                for line in recent[-lines:] if lines else []:
                    print(line, flush=True)
                waiting = False
            for line in handle:
                if marker in line:
                    print(line[line.index(marker):].rstrip(), flush=True)
            time.sleep(interval)
    finally:
        if handle is not None:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18020')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('--metrics', action='store_true', help='use the aggregate Prometheus display instead of request logs')
    parser.add_argument('--log', type=Path, default=Path(os.environ.get('HYPERQWEN_LOG_FILE', Path(__file__).resolve().parents[1] / 'logs/turing.log')))
    parser.add_argument('--lines', type=int, default=30, help='recent request progress lines to show initially')
    parser.add_argument('--once', action='store_true', help='show totals since server startup and exit')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    if args.lines < 0:
        parser.error('--lines must be non-negative')
    if not args.metrics and not args.once:
        follow_log(args.log, args.interval, args.lines)
        return 0
    print('HyperQwen speed monitor — Ctrl+C exits; server stays running.', flush=True)
    print('Prefill/decode summaries appear after completion; multiple completions are grouped.\n'
          'Live output is interval aggregate throughput, not per-request decode speed.', flush=True)
    previous = completed = None
    previous_time = None
    failed = False
    while True:
        try:
            current = read_metrics(args.url)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if not failed:
                print(f'{stamp()} server unavailable: {exc}', flush=True)
            if args.once:
                return 1
            failed = True
            time.sleep(args.interval)
            continue
        now = time.monotonic()
        if failed:
            print(f'{stamp()} server reconnected', flush=True)
            previous = completed = None
            failed = False
        if previous is not None and (
            current.get('vllm:generation_tokens_total', 0) < previous.get('vllm:generation_tokens_total', 0)
            or current.get('vllm:request_prompt_tokens_count', 0) < completed.get('vllm:request_prompt_tokens_count', 0)
        ):
            print(f'{stamp()} server counters reset', flush=True)
            previous = completed = None
        if args.once:
            summary = completed_summary({}, current)
            print('Since server startup: ' + (summary or 'no completed timing samples'), flush=True)
            return 0
        if previous is None:
            print(f'{stamp()} connected — waiting for activity', flush=True)
            completed = current
        else:
            summary = completed_summary(completed, current)
            if summary:
                print(f'{stamp()} {summary}', flush=True)
                completed = current
            generated = max(0, current.get('vllm:generation_tokens_total', 0) - previous.get('vllm:generation_tokens_total', 0))
            running = current.get('vllm:num_requests_running', 0)
            waiting = current.get('vllm:num_requests_waiting', 0)
            if generated or running or waiting:
                rate = generated / (now - previous_time)
                kv = 100 * current.get('vllm:kv_cache_usage_perc', 0)
                print(f'{stamp()} live output={rate:.1f} tok/s | running={int(running)} '
                      f'waiting={int(waiting)} KV={kv:.1f}%', flush=True)
        previous, previous_time = current, now
        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nMonitor stopped.')
