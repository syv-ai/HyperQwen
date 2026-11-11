#!/usr/bin/env python3
"""Linux CPU-only fast-shard link recovery, with an offline Hub snapshot."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
SHARD = "model-00001-of-00007.safetensors"
FAST_SHARD = "model-00007-of-00007.safetensors"
KEYS = ["lm_head.weight_packed", "model.embed_tokens.weight_packed",
        "mtp.layers.0.mlp.down_proj.weight_packed", "mtp.draft_lm_head.weight_packed"]


def tensor_file():
    header = {k: {"dtype": "I32", "shape": [16], "data_offsets": [64*i, 64*(i+1)]}
              for i, k in enumerate(KEYS)}
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    return struct.pack("<Q", len(data)) + data + bytes(256)


class FastLinkTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base, self.fast = self.root / "model", self.root / "model-fast"
        self.base.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (self.base / name).write_text("{}", encoding="utf-8")
        (self.base / SHARD).write_bytes(tensor_file())
        (self.base / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {k: SHARD for k in KEYS}}), encoding="utf-8")
        (self.base / "mtp_draft_vocab_ids.pt").write_bytes(b"ids")
        self.original = {p.name: p.read_bytes() for p in self.base.iterdir()}
        hub = self.root / "hub"
        hub.mkdir()
        for name, data in ((FAST_SHARD, tensor_file()),
                           ("model_extra_tensors.safetensors", tensor_file()),
                           ("mtp_draft_vocab_ids.pt", b"ids"), ("config.json", b"{}"),
                           ("model.safetensors.index.json", json.dumps({"weight_map": {
                               k: FAST_SHARD if k == KEYS[-1] else SHARD for k in KEYS}}).encode())):
            (hub / name).write_bytes(data)
        stub = self.root / "stub"
        stub.mkdir()
        (stub / "huggingface_hub.py").write_text(
            "import os\ndef snapshot_download(*args, **kwargs):\n"
            "    return os.environ['HUB_FIXTURE']\n", encoding="utf-8")
        self.env = dict(os.environ, PYTHONPATH=str(stub), HUB_FIXTURE=str(hub),
                        HF_HUB_OFFLINE="1")
        self.fetch = REPO / "prepare/fetch_fast_variant.py"

    def run_fetch(self):
        result = subprocess.run([sys.executable, str(self.fetch), str(self.base), str(self.fast)],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def assert_ready(self, expected=""):
        source = (REPO / "docker/prepare.sh").read_text(encoding="utf-8")
        block = source.split('python - "$BASE" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0]
        result = subprocess.run([sys.executable, "-", str(self.base)], input=block,
                                env=dict(self.env, FAST_VARIANT="1", DFLASH2="0"),
                                capture_output=True, text=True)
        self.assertEqual((result.returncode, result.stderr), (0, ""))
        self.assertEqual(result.stdout.strip(), expected)

    def damage_then_repair_base(self):
        # A previous in-place write tears the shared inode. An atomic repair
        # replaces only the base name, leaving the fast link on the torn inode.
        self.run_fetch()
        (self.fast / SHARD).write_bytes(b"torn shard")
        self.old_link = self.root / "old-link"
        os.link(self.fast / SHARD, self.old_link)
        replacement = self.base / "repair.tmp"
        replacement.write_bytes(self.original[SHARD])
        os.replace(replacement, self.base / SHARD)
        self.assert_ready("fast")

    def assert_repaired(self):
        self.assert_ready()
        self.assertTrue(os.path.samefile(self.base / SHARD, self.fast / SHARD))
        self.assertEqual({p.name: p.read_bytes() for p in self.base.iterdir()}, self.original)
        self.assertEqual(self.old_link.read_bytes(), b"torn shard")

    def test_fresh_assembly_and_rerun_keep_shared_inode(self):
        self.run_fetch()
        self.assert_ready()
        self.assertTrue(os.path.samefile(self.base / SHARD, self.fast / SHARD))
        before = (self.fast / SHARD).stat().st_ino
        self.run_fetch()
        self.assertEqual((self.fast / SHARD).stat().st_ino, before)
        self.assertEqual({p.name: p.read_bytes() for p in self.base.iterdir()}, self.original)

    def test_repaired_base_replaces_damaged_fast_link(self):
        self.damage_then_repair_base()
        self.run_fetch()
        self.assert_repaired()

    def test_interrupted_link_publication_can_resume(self):
        self.damage_then_repair_base()
        code = '''import os, runpy, sys
sys.path.insert(0, sys.argv[1])
import atomic_publish
publish = atomic_publish.publish
def interrupt(tmp, dst):
    if os.path.basename(dst) == os.environ['STOP_SHARD']:
        raise RuntimeError('simulated interruption before shard publication')
    return publish(tmp, dst)
atomic_publish.publish = interrupt
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        result = subprocess.run(
            [sys.executable, "-c", code, str(REPO / "prepare"), str(self.fetch),
             str(self.base), str(self.fast)], env=dict(self.env, STOP_SHARD=SHARD),
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simulated interruption before shard publication", result.stderr)
        self.assertEqual((self.fast / SHARD).read_bytes(), b"torn shard")
        self.assertEqual((self.base / SHARD).read_bytes(), self.original[SHARD])
        self.assertTrue(os.path.samefile(self.base / SHARD, self.fast / (SHARD + ".tmp")))
        self.run_fetch()
        self.assert_repaired()
        self.assertFalse((self.fast / (SHARD + ".tmp")).exists())


if __name__ == "__main__":
    unittest.main()
