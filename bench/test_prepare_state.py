#!/usr/bin/env python3
"""CPU-only checks that docker/prepare.sh's state() survives incomplete models.

Older prepare scripts wrote the index, config.json and the shards in place, so a kill
could leave any of them truncated. state() then died with JSONDecodeError, and
docker/entrypoint.sh, which runs prepare.sh under `set -e` before every start, stopped
every later start. state() must answer `download` for each torn file (`hf download`
fetches files that differ from the Hub again) and must never crash. Fast-variant
checks also cover invalid index maps and offline recovery through the real fetcher.
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


class FastVariantStateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base = self.root / "model"
        self.fast = self.root / "model-fast"
        self.base.mkdir()
        self.fast.mkdir()
        self.base_shard = "model-00001-of-00007.safetensors"
        self.fast_shard = "model-00007-of-00007.safetensors"
        self.index = {"weight_map": {k: self.base_shard for k in KEYS}}
        self.fast_index = {"weight_map": {
            k: self.fast_shard if k == KEYS[-1] else self.base_shard for k in KEYS}}
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (self.base / name).write_text("{}", encoding="utf-8")
        (self.base / "mtp_draft_vocab_ids.pt").write_bytes(b"ids")
        (self.base / self.base_shard).write_bytes(shard(256))
        (self.base / "model.safetensors.index.json").write_text(
            json.dumps(self.index), encoding="utf-8")
        os.link(self.base / self.base_shard, self.fast / self.base_shard)
        (self.fast / self.fast_shard).write_bytes(shard(256))
        self.fast_index_path = self.fast / "model.safetensors.index.json"
        self.fast_index_path.write_text(json.dumps(self.fast_index), encoding="utf-8")

    def state(self, fast="1"):
        source = (REPO / "docker/prepare.sh").read_text(encoding="utf-8")
        block = source.split('python - "$BASE" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
        result = subprocess.run(
            [sys.executable, "-", str(self.base)], input=block, text=True,
            capture_output=True, env=dict(os.environ, FAST_VARIANT=fast, DFLASH2="0"))
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        return result.stdout.strip()

    def test_complete_variant_and_explicit_opt_out(self):
        self.assertEqual(self.state(), "")
        self.fast_index_path.write_text('{"weight_map": null}', encoding="utf-8")
        before = self.fast_index_path.read_bytes()
        self.assertEqual(self.state(fast="0"), "")
        self.assertEqual(self.fast_index_path.read_bytes(), before)

    def test_invalid_index_requests_fast_repair(self):
        cases = [None, [], {}, {"weight_map": {}}, {"weight_map": None},
                 {"weight_map": []}, {"weight_map": [self.fast_shard]},
                 {"weight_map": 42}, {"weight_map": self.fast_shard}]
        for index in cases:
            with self.subTest(index=index):
                self.fast_index_path.write_text(json.dumps(index), encoding="utf-8")
                self.assertEqual(self.state(), "fast")

    def test_missing_or_torn_files_request_fast_repair(self):
        cases = [("missing index", "model.safetensors.index.json", None),
                 ("torn index", "model.safetensors.index.json", b'{"weight_map":'),
                 ("missing linked shard", self.base_shard, None),
                 ("torn downloaded shard", self.fast_shard, shard(64)),
                 ("invalid downloaded header", self.fast_shard, b"invalid")]
        for label, name, data in cases:
            with self.subTest(case=label):
                path = self.fast / name
                original = path.read_bytes()
                try:
                    if data is None:
                        path.unlink()
                    else:
                        path.write_bytes(data)
                    self.assertEqual(self.state(), "fast")
                finally:
                    path.write_bytes(original)

    def test_fetcher_repairs_invalid_index_offline(self):
        # Run the real fetcher, substituting only Hub acquisition with a tiny
        # local snapshot. No network, model download or third-party library.
        hub = self.root / "hub"
        hub.mkdir()
        for name, data in ((self.fast_shard, shard(256)),
                           ("model_extra_tensors.safetensors", shard(256)),
                           ("mtp_draft_vocab_ids.pt", b"ids"),
                           ("config.json", b"{}"),
                           ("model.safetensors.index.json",
                            json.dumps(self.fast_index).encode("utf-8"))):
            (hub / name).write_bytes(data)
        stub = self.root / "stub"
        stub.mkdir()
        (stub / "huggingface_hub.py").write_text(
            "import os\n"
            "def snapshot_download(repo_id, **kwargs):\n"
            "    assert repo_id == 'syvai/qwen3.8-27b-3090-fast-variant'\n"
            "    return os.environ['HUB_FIXTURE']\n", encoding="utf-8")
        original = {f.name: f.read_bytes() for f in self.base.iterdir()}
        for mapping in (None, {}):
            with self.subTest(weight_map=mapping):
                self.fast_index_path.write_text(
                    json.dumps({"weight_map": mapping}), encoding="utf-8")
                self.assertEqual(self.state(), "fast")
                result = subprocess.run(
                    [sys.executable, str(REPO / "prepare/fetch_fast_variant.py"),
                     str(self.base), str(self.fast)], capture_output=True, text=True,
                    env=dict(os.environ, PYTHONPATH=str(stub), HUB_FIXTURE=str(hub),
                             HF_HUB_OFFLINE="1"))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.state(), "")
                self.assertEqual(
                    {f.name: f.read_bytes() for f in self.base.iterdir()}, original)


if __name__ == "__main__":
    unittest.main()
