#!/usr/bin/env python3
"""Live terminal dashboard for a HyperQwen/vLLM server, in Project Maya's look.

Draws the dashboard from Project Maya (https://github.com/mw00/project-maya, MIT, Peasant Smith and the Project Maya
contributors; tools/setup_dash.py: big tok/s, prefill, context, GPUs, hardware, recent requests) from vLLM's
Prometheus endpoint, nvidia-smi and the "[hyperqwen]" request log. Maya's code is not copied into this repo: it is
imported from your own Project Maya checkout. It never sends requests.

Needs Textual and a Project Maya checkout. The checkout is MAYA_DIR, else ../project-maya next to this repo, else
~/project-maya. It runs with the current Python if that has Textual, else with the checkout's `.venv`. With neither
(for example in the Docker image) it falls back to a plain curses display.

    python3 scripts/tui_turing.py            # q or Ctrl+C leaves; the server keeps running
Env: HYPERQWEN_URL (default http://127.0.0.1:18020), VLLM_API_KEY, HYPERQWEN_LOG_FILE, MAYA_DIR
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_maya() -> Path:
    candidates = [os.environ.get("MAYA_DIR"), HERE.parent.parent / "project-maya", Path.home() / "project-maya"]
    for c in candidates:
        if c and (Path(c) / "tools" / "setup_dash.py").is_file():
            return Path(c)
    return Path(candidates[0] or candidates[1])


MAYA = find_maya()
URL = os.environ.get("HYPERQWEN_URL", "http://127.0.0.1:18020")
KEY = os.environ.get("VLLM_API_KEY", "")
LOG = os.environ.get("HYPERQWEN_LOG_FILE") or str(HERE.parent / "logs" / "turing.log")


def fallback() -> None:
    sys.path.insert(0, str(HERE))
    from tui_turing_stdlib import main
    main()
    raise SystemExit(0)


try:
    import rich      # noqa: F401
    import textual   # noqa: F401
except ModuleNotFoundError:
    venv_python = MAYA / ".venv" / "bin" / "python"
    if (MAYA / "tools" / "setup_dash.py").is_file() and venv_python.exists() and not os.environ.get("_HQ_REEXEC"):
        os.environ["_HQ_REEXEC"] = "1"
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    fallback()

tools = MAYA / "tools"
if not (tools / "setup_dash.py").is_file():
    fallback()
sys.path.insert(0, str(tools))
sys.path.insert(0, str(HERE))

from textual.app import App                                              # noqa: E402
from setup_dash import Dashboard                                         # noqa: E402
from setup_look import ACCENT_TEXT, EDGE, LINE, MUTED, THEME, TINT       # noqa: E402
from turing_metrics import Source                                        # noqa: E402


class VllmDashboard(Dashboard):
    """Maya's dashboard without the expert-tier box (vLLM has no expert cache): the throughput box takes the row."""

    DEFAULT_CSS = """
    VllmDashboard #dash-tiers { display: none; }
    VllmDashboard #dash-card { display: none; }
    VllmDashboard #dash-speed { margin-right: 0; }
    """

    def fill(self, m: dict) -> None:
        # Maya sizes the speed box as half the screen (it shares the row); here it has the whole row.
        self.screen.add_class("narrow")
        super().fill(m)
        self.query_one("#dash-tiers").display = False
        self.query_one("#dash-card").display = False


class Turing(App):
    serving = {"state": "ready"}
    BINDINGS = [("q", "quit", "Quit")]

    def get_css_variables(self):
        return {**super().get_css_variables(), "maya-edge": EDGE, "maya-line": LINE, "maya-tint": TINT,
                "maya-muted": MUTED, "maya-accent-text": ACCENT_TEXT}

    def compose(self):
        yield VllmDashboard(id="dash")

    def on_mount(self):
        self.register_theme(THEME)
        self.theme = "maya"
        self.title = "HyperQwen · Turing"
        dash = self.query_one(VllmDashboard)
        dash.focus()
        source = Source(URL, KEY, LOG)

        def poll():
            while True:
                try:
                    dash.metrics, dash.fails = source.snapshot(), 0
                except Exception:                          # noqa: BLE001 - server down/restarting: keep going
                    dash.fails += 1
                time.sleep(1)
        threading.Thread(target=poll, name="turing-poll", daemon=True).start()


if __name__ == "__main__":
    Turing().run()
