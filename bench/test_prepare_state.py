#!/usr/bin/env python3
"""CPU-only checks that docker/prepare.sh's state() survives torn files (#195).

Older prepare scripts wrote the index, config.json and the shards in place, so a kill
could leave any of them truncated. state() then died with JSONDecodeError, and
docker/entrypoint.sh, which runs prepare.sh under `set -e` before every start, stopped
every later start. state() must answer `download` for each torn file (`hf download`
fetches files that differ from the Hub again) and must never crash.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
SHARD = "model-00001-of-00001.safetensors"
KEYS = ["lm_head.weight_packed", "model.embed_tokens.weight_packed",
        "mtp.layers.0.mlp.down_proj.weight_packed", "mtp.draft_lm_head.weight_packed"]


def shard(payload):
    """A safetensors file with four 64-byte tensors and `payload` bytes of data."""
    header = {k: {"dtype": "I32", "shape": [16], "data_offsets": [64 * i, 64 * i + 64]}
              for i, k in enumerate(KEYS)}
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


class PrepareStateTests(unittest.TestCase):
    def test_torn_files(self):
        source = (REPO / "docker/prepare.sh").read_text(encoding="utf-8")
        block = source.split('python - "$BASE" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
        index = json.dumps({"metadata": {}, "weight_map": {k: SHARD for k in KEYS}})
        # (case, file to overwrite, its bytes, expect download); None deletes the file
        cases = [("prepared dir", None, None, False),
                 ("truncated index", "model.safetensors.index.json", index[:40].encode(), True),
                 ("index without weight_map", "model.safetensors.index.json", b'{"metadata": {}}', True),
                 ("truncated config", "config.json", b'{"architectures": [', True),
                 ("truncated shard", SHARD, shard(64), True),
                 ("unparsable shard header", SHARD, struct.pack("<Q", 32) + b"not a json header", True),
                 ("missing shard", SHARD, None, True)]
        for name, path, data, download in cases:
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                model = Path(tmp)
                for f, content in (("tokenizer.json", b"{}"), ("tokenizer_config.json", b"{}"),
                                   ("config.json", b"{}"), ("mtp_draft_vocab_ids.pt", b"ids"),
                                   ("model.safetensors.index.json", index.encode()), (SHARD, shard(256))):
                    (model / f).write_bytes(content)
                if path and data is None:
                    (model / path).unlink()
                elif path:
                    (model / path).write_bytes(data)
                r = subprocess.run([sys.executable, "-", tmp], input=block, text=True, capture_output=True,
                                   env=dict(os.environ, FAST_VARIANT="0", DFLASH2="0"))
                # A crash must not pass as a `download`.
                self.assertEqual((r.returncode, r.stderr), (0, ""))
                self.assertEqual(r.stdout.strip() == "download", download, r.stdout)


if __name__ == "__main__":
    unittest.main()
