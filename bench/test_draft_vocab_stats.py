"""Offline CPU regressions for build_draft_vocab.py's read-only statistics.

Run: venv/bin/python bench/test_draft_vocab_stats.py
Needs torch, transformers, tokenizers and safetensors; no model download or GPU.
Exit 0 when all checks pass, 1 on failure.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers

REPO = Path(__file__).resolve().parents[1]


class DraftVocabStatsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        tokenizer = Tokenizer(models.WordLevel(
            {"[UNK]": 0, "a": 1, "b": 2, "Ж": 3}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.save(str(self.model / "tokenizer.json"))
        (self.model / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]",
        }), encoding="utf-8")
        self.ids = self.root / "ids.json"
        self.ids.write_text("[1, 3]", encoding="utf-8")
        self.corpus = self.root / "corpus.jsonl"
        self.corpus.write_text(
            (json.dumps({"text": "a b Ж"}) + "\n") * 12, encoding="utf-8")

    def snapshot(self):
        return {str(p.relative_to(self.model)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.model.rglob("*") if p.is_file()}

    def run_script(self, *args, readonly=True):
        before = self.snapshot()
        result = subprocess.run(
            [sys.executable, str(REPO / "prepare/build_draft_vocab.py"),
             str(self.model), *map(str, args)], capture_output=True, text=True,
            encoding="utf-8", timeout=60, env={**os.environ,
                "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUTF8": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if readonly:
            self.assertEqual(before, self.snapshot(), "statistics modified the model")
        return result.stdout

    def test_ids_stats_counts_corpus_and_held_out_coverage(self):
        for flags in (("--ids", self.ids, "--stats", "--corpus", self.corpus),
                      ("--corpus", self.corpus, "--ids", self.ids, "--stats")):
            with self.subTest(flags=flags):
                out = self.run_script(*flags)
                self.assertIn("corpus tokens: 36", out)
                self.assertIn("held-out token coverage 66.67% (4/6 tokens)", out)
                self.assertRegex(out, r"(?m)^cyrillic\s+1\s+1\s+1\s+100\.0%\s+50\.0%$")

    def test_stats_without_ids_reports_scripts_with_no_selection(self):
        out = self.run_script("--stats")
        for script in ("han", "kana", "hangul", "cjk_punct", "cyrillic", "latin_ext", "CJK total"):
            self.assertRegex(out, rf"(?m)^{script}\s+\d+\s+\d+\s+-\s+-\s+-$")
        self.assertRegex(out, r"(?m)^cyrillic\s+1\s+1\s+-\s+-\s+-$")
        self.assertNotIn("held-out token coverage", out)

    def test_empty_id_selection_has_zero_coverage(self):
        self.ids.write_text("[]", encoding="utf-8")
        out = self.run_script("--ids", self.ids, "--stats", "--corpus", self.corpus)
        self.assertIn("held-out token coverage 0.00% (0/6 tokens)", out)
        self.assertRegex(out, r"(?m)^cyrillic\s+1\s+1\s+0\s+0\.0%\s+0\.0%$")

    def test_no_held_out_tokens_is_not_reported_as_zero_coverage(self):
        self.corpus.write_text('{"text": ""}\n{"text": "a b"}\n', encoding="utf-8")
        out = self.run_script("--ids", self.ids, "--stats", "--corpus", self.corpus)
        self.assertIn("corpus tokens: 2", out)
        self.assertIn("held-out token coverage n/a (no held-out tokens)", out)

    def test_stats_and_script_union_still_write_only_variant_list(self):
        self.ids.write_text("[1]", encoding="utf-8")
        variant = self.root / "variant.json"
        out = self.run_script("--ids", self.ids, "--stats", "--corpus", self.corpus,
                              "--add-scripts", "cyrillic", "--out", variant)
        self.assertIn("held-out token coverage 33.33% (2/6 tokens)", out)
        self.assertEqual(json.loads(variant.read_text()), [1, 3])

    def test_plain_ids_build_still_ignores_corpus(self):
        import torch
        from safetensors.torch import load_file, save_file

        tensors = {"lm_head.weight_packed": torch.arange(32, dtype=torch.int32).reshape(4, 8),
                   "lm_head.weight_scale": torch.ones(4, 1, dtype=torch.bfloat16),
                   "lm_head.weight_shape": torch.tensor([4, 64])}
        save_file(tensors, self.model / "head.safetensors")
        (self.model / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "head.safetensors" for name in tensors}}), encoding="utf-8")
        out = self.run_script("--ids", self.ids, "--corpus", self.root / "missing.txt", readonly=False)
        self.assertIn("corpus tokens: 0", out)
        self.assertEqual(torch.load(self.model / "mtp_draft_vocab_ids.pt").tolist(), [1, 3])
        extra = load_file(self.model / "model_extra_tensors.safetensors")
        self.assertTrue(torch.equal(extra["mtp.draft_lm_head.weight_packed"],
                                    tensors["lm_head.weight_packed"][[1, 3]]))


if __name__ == "__main__":
    unittest.main()
