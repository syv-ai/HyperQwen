"""Choose one explicit vLLM source tree for reproducible benchmark/calibration input."""
import importlib.util
import os
from pathlib import Path

CHECKOUT_SOURCE = Path(__file__).resolve().parents[1] / "venv/lib/python3.12/site-packages/vllm"


def source_files(subdir, recursive=False):
    """Prefer CODE_CORPUS_DIR, then the historical checkout tree, then the active install.

    CODE_CORPUS_DIR names the vllm package root, not its parent or v1 directory.
    An invalid explicit selection is an error; it must never select a different tree.
    """
    configured = os.environ.get("CODE_CORPUS_DIR")
    if configured:
        root = Path(configured).expanduser()
    elif CHECKOUT_SOURCE.is_dir():
        root = CHECKOUT_SOURCE
    else:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("No vLLM code source; set CODE_CORPUS_DIR to a fixed vllm package directory")
        root = Path(next(iter(spec.submodule_search_locations)))
    root = root.resolve()
    directory = root / subdir
    files = sorted(directory.glob("**/*.py" if recursive else "*.py"))
    if not files:
        raise RuntimeError(f"No vLLM code corpus under {directory}; set CODE_CORPUS_DIR to a fixed vllm package directory")
    return root, files
