"""How the bench/ scripts reach the server: the key, the URL, the request, the stream and /metrics.

Every script runs as `python bench/<script>.py`, which puts bench/ on sys.path, so `import harness` needs no
path setup. Stdlib only: the scripts run on the venv's bare Python.
"""
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def client_key():
    """resolve_client_key in resolve_api_key.sh: OPENAI_API_KEY, else VLLM_API_KEY, else api_key.txt, else
    "EMPTY" (a server that bound no key ignores it). test_harness.py runs both and compares."""
    for name in ("OPENAI_API_KEY", "VLLM_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    path = os.path.join(REPO, "api_key.txt")
    if os.path.isfile(path):
        with open(path, newline="") as f:
            return f.read().rstrip("\n")
    return "EMPTY"


def base_url():
    """The server root: VLLM_API if set, else http://127.0.0.1:${PORT:-18020}. A trailing /v1 on VLLM_API is
    dropped, because five scripts documented VLLM_API with it."""
    api = os.environ.get("VLLM_API", "").rstrip("/")
    if api:
        return api.removesuffix("/v1")
    return "http://127.0.0.1:" + os.environ.get("PORT", "18020")


def request(path, payload=None):
    """A request to base_url() + path with the key; a payload makes it a JSON POST."""
    headers = {"Authorization": "Bearer " + client_key()}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    return urllib.request.Request(base_url() + path, data=data, headers=headers)


def post(path, payload, timeout=1200):
    """JSON POST to base_url() + path; returns the decoded JSON body."""
    with urllib.request.urlopen(request(path, payload), timeout=timeout) as r:
        return json.load(r)


def stream(path, payload, timeout=1800):
    """POST a streaming request and yield each SSE data frame, decoded, until [DONE]. Lines that are not data
    (the blank separators, a `: keep-alive` comment) are skipped. The caller times the frames itself."""
    with urllib.request.urlopen(request(path, payload), timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            yield json.loads(body)


def samples(timeout=30):
    """GET /metrics as a list of (name, labels, value), one per sample line. labels is the raw "{...}" text,
    "" when the line has none. Label values may hold spaces, so the labels end at the last "}"."""
    with urllib.request.urlopen(request("/metrics"), timeout=timeout) as r:
        text = r.read().decode()
    out = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        brace, space = line.find("{"), line.find(" ")
        if 0 <= brace < space:
            end = line.rindex("}") + 1
            name, labels, rest = line[:brace], line[brace:end], line[end:]
        else:
            name, labels, rest = line[:space], "", line[space:]
        out.append((name, labels, float(rest.split()[0])))
    return out


def metrics(*names, timeout=30):
    """{name: value summed over its label sets, so over engines} for each of names present on /metrics. Names
    match exactly, so a counter's <name>_created line never adds to <name>_total."""
    out = {}
    for name, _, value in samples(timeout):
        if name in names:
            out[name] = out.get(name, 0.0) + value
    return out


SPEC = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total")


def spec():
    """(drafts, accepted tokens) so far. Over a window, tokens/step is 1 + accepted delta / drafts delta."""
    m = metrics(*SPEC)
    return m.get(SPEC[0], 0.0), m.get(SPEC[1], 0.0)


if __name__ == "__main__":
    # The bash scripts call `python3 bench/harness.py spec` for "drafts accepted". With no server it prints
    # nothing on stdout, as their `curl -s` did, and one line on stderr instead of a traceback.
    if sys.argv[1:] != ["spec"]:
        sys.exit("usage: harness.py spec")
    try:
        print(*spec())
    except OSError as e:
        sys.exit(f"harness.py spec: {e}")
