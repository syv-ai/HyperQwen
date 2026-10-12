"""Verify the CUDA source diff against the flash-attention commit pinned in PIN.

python3 fa2/check_upstream.py --fa-dir <git checkout of vllm-project/flash-attention at FA_COMMIT>
(or set FA_DIR). --emit prints the diff that fa2-fp8kv.patch must hold.
"""
import argparse
from dataclasses import dataclass
import difflib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

SOURCE = Path("csrc/flash_attn/src")


@dataclass(frozen=True)
class SourceComparison:
    root: Path
    fa_dir: Path
    revision: str

    def mappings(self):
        headers = tuple((SOURCE / path.name, path)
                        for path in sorted((self.root / "headers").iterdir())
                        if path.is_file())
        additions = tuple((SOURCE / name, self.root / name)
                          for name in ("fp8_attn.cu", "prefill.cu"))
        return headers + additions

    def original(self, path):
        # The two standalone extension entrypoints are new project code.
        if path.name in ("fp8_attn.cu", "prefill.cu"):
            return b""
        return subprocess.run(
            ["git", "-C", str(self.fa_dir),
             "show", f"{self.revision}:{path.as_posix()}"],
            check=True, capture_output=True,
        ).stdout

    def patch(self):
        changes = []
        for upstream, local in self.mappings():
            original, modified = self.original(upstream), local.read_bytes()
            if original == modified:
                continue
            name = upstream.as_posix()
            changes.append(f"diff --git a/{name} b/{name}\n")
            if not original:
                changes.append("new file mode 100644\n")
            changes.extend(difflib.unified_diff(
                original.decode("utf-8").splitlines(keepends=True),
                modified.decode("utf-8").splitlines(keepends=True),
                fromfile=f"a/{name}" if original else "/dev/null",
                tofile=f"b/{name}",
            ))
        return "".join(changes)

    def verified(self):
        head = subprocess.run(
            ["git", "-C", str(self.fa_dir), "rev-parse", "HEAD"],
            check=True, capture_output=True, encoding="utf-8",
        ).stdout.strip()
        if head != self.revision:
            raise ValueError(f"Upstream must be {self.revision}; found {head}")
        patch = self.root / "fa2-fp8kv.patch"
        if patch.read_text(encoding="utf-8") != self.patch():
            raise ValueError("CUDA sources differ from fa2/fa2-fp8kv.patch; regenerate it with --emit")
        # Apply in a disposable tree. The checkout remains an untouched baseline.
        with tempfile.TemporaryDirectory(prefix="fa2-upstream-check-") as directory:
            tree = Path(directory)
            subprocess.run(["git", "init", "--quiet", directory], check=True)
            for mapped, _ in self.mappings():
                original = self.original(mapped)
                if original:
                    target = tree / mapped
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original)
            for flags in (["--check"], []):
                subprocess.run(["git", "apply", *flags, str(patch)], cwd=tree,
                               check=True, capture_output=True, encoding="utf-8")
            for mapped, local in self.mappings():
                if (tree / mapped).read_bytes() != local.read_bytes():
                    raise ValueError(f"Patched source differs: {mapped}")
        return f"PASS: patch applies to {self.revision}; all {len(self.mappings())} mapped files match byte-for-byte"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="Print the current CUDA source diff")
    parser.add_argument("--fa-dir", type=Path, default=os.environ.get("FA_DIR"),
                        help="git checkout of the flash-attention repo at FA_COMMIT (default: $FA_DIR)")
    args = parser.parse_args()
    if args.fa_dir is None:
        parser.error("pass --fa-dir or set FA_DIR")
    root = Path(__file__).resolve().parent
    pin = dict(line.split("=", 1) for line in (root / "PIN").read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#"))
    comparison = SourceComparison(root, Path(args.fa_dir), pin["FA_COMMIT"])
    print(comparison.patch(), end="") if args.emit else print(comparison.verified())
