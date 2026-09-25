"""Translate a Qwen3.8 chat template's reasoning-effort vocabulary so client
supplied effort levels can no longer 400 the server.

The shipped chat_template.jinja accepts exactly xhigh/medium/low and defaults
to xhigh. OpenAI-protocol clients speak the gpt-5 vocabulary --
none/minimal/low/medium/high/xhigh/max -- so `minimal` (and `high`, `max`)
raise jinja2.TemplateError inside the template and vLLM surfaces that as a
400 Bad Request on every request that carries one (docs/gotchas.md 58).

This rewrites the effort block in place to:

  - map only the names the template does not know: minimal -> low,
    high/max -> xhigh,
  - leave every other value falling through unchanged, so the template's own
    levels (low/medium/xhigh) keep their exact behaviour and an omitted effort
    keeps the template default (xhigh),
  - drop the raise, so an unknown value no longer 400s. It then produces no
    reasoning instruction, the same outcome as `medium` -- nothing is silently
    rewritten to a stronger level.

Idempotent: the `{#- effort-translation v2 ... -#}` marker short-circuits
re-runs, so prepare.sh can call this on every boot, and a re-download that
clobbers chat_template.jinja is re-translated on the next prepare. A template
carrying the older v1 block (default and fallback both `medium`) is upgraded in
place, so a model dir prepared with v1 does not keep that default. A template
whose effort block matches no known shape is left alone with a warning: this
runs under `set -e` after the download and requantisation, so it must never
fail prepare over a ready model directory.

The marker makes a re-run skip the file, so a truncated template that kept the marker
would stay broken. The rewrite is therefore atomic (a temp file and a rename, see
prepare/atomic_publish.py): a kill during the write leaves the old template (#195).

Usage: python prepare/translate_chat_template.py MODEL_DIR [MODEL_DIR ...]
"""

import sys
from pathlib import Path

from atomic_publish import write_text

MARKER_V2 = "{#- effort-translation v2: managed by prepare/translate_chat_template.py"
MARKER_V1 = "{#- effort-translation v1: managed by prepare/translate_chat_template.py"

# The shipped shape (chat_template.jinja lines 47-50): the default line, the
# vocabulary guard, the raise, and its endif. Matched as a structural span
# (first line located by anchor, next three by shape), not as one brittle blob.
ANCHOR = "set resolved_reasoning_effort = reasoning_effort|default("
SHAPES = (
    "if resolved_reasoning_effort not in",
    "raise_exception(",
    "endif",
)

REPLACEMENT = """\
    {#- effort-translation v2: managed by prepare/translate_chat_template.py.
        Map only the gpt-5 names the template does not know (minimal -> low,
        high/max -> xhigh); every other value falls through unchanged, and an
        omitted effort keeps the template default (xhigh). docs/gotchas.md 58. -#}
    {%- set effort_aliases = {'minimal': 'low', 'high': 'xhigh', 'max': 'xhigh'} %}
    {%- set requested_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 'xhigh' %}
    {%- set resolved_reasoning_effort = effort_aliases.get(requested_effort, requested_effort) %}"""

# The v1 block written by the first cut of this script: same three statements,
# default and fallback both `medium`. Upgraded rather than left in place --
# otherwise a dir prepared with v1 keeps a default that re-baselines every
# quality number measured at xhigh. Matched by fragment, so the comment's line
# wrapping is not part of the contract.
V1_FRAGMENTS = (
    "'minimal': 'low', 'low': 'low', 'medium': 'medium'",
    "else 'medium'",
    "effort_aliases.get(requested_effort, 'medium')",
)


def translate(path: Path) -> str:
    """Rewrite one chat_template.jinja.

    Returns translated | upgraded | already | missing | foreign.
    """
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8")
    if MARKER_V2 in text:
        return "already"
    lines = text.splitlines(keepends=True)

    # v1 -> v2: replace from the v1 marker line through its resolved line.
    start = next((i for i, l in enumerate(lines) if MARKER_V1 in l), None)
    if start is not None:
        end = next((j for j in range(start, min(start + 8, len(lines)))
                    if "effort_aliases.get(requested_effort, 'medium')" in lines[j]), None)
        if end is None:
            return "foreign"
        block = "".join(lines[start:end + 1])
        if not all(frag in block for frag in V1_FRAGMENTS):
            return "foreign"
        lines[start:end + 1] = [REPLACEMENT + "\n"]
        write_text(path, "".join(lines))
        return "upgraded"

    # pristine -> v2: the shipped block, located structurally.
    idx = next((i for i, l in enumerate(lines) if ANCHOR in l), None)
    if idx is None:
        return "foreign"
    span = lines[idx + 1 : idx + 1 + len(SHAPES)]
    if len(span) < len(SHAPES):
        return "foreign"
    if not all(shape in span[i] for i, shape in enumerate(SHAPES)):
        return "foreign"
    lines[idx : idx + 1 + len(SHAPES)] = [REPLACEMENT + "\n"]
    write_text(path, "".join(lines))
    return "translated"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: translate_chat_template.py MODEL_DIR [MODEL_DIR ...]")
    failed = []
    for arg in sys.argv[1:]:
        path = Path(arg.rstrip("/")) / "chat_template.jinja"
        try:
            result = translate(path)
        except OSError as exc:
            result = f"write failed ({exc.strerror})"
        if result == "foreign":
            print(f"translate_chat_template: {path}: WARN effort block matches no known "
                  f"shape, left unchanged (template changed upstream; TRANSLATE_EFFORT=0 skips)")
        elif result.startswith("write failed"):
            print(f"translate_chat_template: {path}: FAIL {result}")
            failed.append(f"{path}: {result}")
        else:
            print(f"translate_chat_template: {path}: {result}")
    if failed:
        sys.exit("translate_chat_template: could not rewrite " + "; ".join(failed))


if __name__ == "__main__":
    main()
