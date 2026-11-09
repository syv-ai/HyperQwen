#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real GDN divergence replay with configurable prefill chunks and frozen token IDs.

Run --variant enabled then default-off with the same --output-dir, each in a
fresh process. Add -dflash to both variants for the actual seven-draft path.
--compare-only checks saved prompt hashes/output tokens without loading a model.
--require-off-miss additionally refuses a result whose baseline already hits.
Needs two free CUDA GPUs, cached model weights and the patched current vLLM.
"""

import argparse
import hashlib
import json
import os
import types
import time
from collections.abc import Mapping
from pathlib import Path


class GDNProbe:
    def checkpoint_probe_install(self):
        model = self.model_runner.get_model()
        found = [
            (name, mod)
            for name, mod in model.named_modules()
            if hasattr(mod, "_store_gdn_prefill_checkpoint")
        ]
        if not found:
            raise RuntimeError("No GDN exporter found")
        name, layer = found[0]
        self._cp_records = []
        self._cp_forwards = []
        self._cp_drafts = 0
        original = layer._store_gdn_prefill_checkpoint
        records = self._cp_records
        forwards = self._cp_forwards

        def observe(this, metadata, *args, **kwargs):
            forwards.append(args[1].shape[1])
            result = original(metadata, *args, **kwargs)
            plan = metadata.checkpoint_plan
            if plan is not None:
                records.append(
                    {
                        "slices": plan.source_slices,
                        "destinations": plan.state_indices.cpu().tolist(),
                        "export_batches": len(plan.export_batches or [plan]),
                    }
                )
            return result

        layer._store_gdn_prefill_checkpoint = types.MethodType(observe, layer)
        speculator = self.model_runner.speculator
        if speculator is not None:
            propose = speculator.propose

            def observe_draft(this, *args, **kwargs):
                self._cp_drafts += 1
                return propose(*args, **kwargs)

            speculator.propose = types.MethodType(observe_draft, speculator)
        fields = (
            "block_size",
            "num_prefill_checkpoint_blocks",
            "prefill_checkpoint_interval",
            "prefill_checkpoint_alignment",
        )

        def describe(spec):
            return {field: getattr(spec, field, None) for field in fields}

        groups = self.model_runner.kv_cache_config.kv_cache_groups
        sizes = [g.kv_cache_spec.block_size for g in groups]
        return {
            "layer": name,
            "backend": layer.gdn_prefill_backend,
            "hash_block_size": self.vllm_config.cache_config.prefix_match_unit
            or min(sizes),
            "speculator": type(speculator).__name__,
            "layer_spec": describe(layer.get_kv_cache_spec(self.vllm_config)),
            "group_specs": [describe(g.kv_cache_spec) for g in groups],
            "retention_interval": self.vllm_config.cache_config.prefix_cache_retention_interval,
        }

    def checkpoint_probe_read(self):
        return list(self._cp_records)

    def checkpoint_probe_forwards(self):
        return list(self._cp_forwards)

    def checkpoint_probe_drafting(self):
        return self._cp_drafts

    def checkpoint_probe_reset(self):
        self._cp_records.clear()
        self._cp_forwards.clear()
        self._cp_drafts = 0
        import torch

        torch.cuda.reset_peak_memory_stats()
        return True

    def checkpoint_probe_memory(self):
        import torch

        return {
            "allocated": torch.cuda.memory_allocated(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }


def compare_results(directory, dflash, require_off_miss):
    suffix = "-dflash" if dflash else ""
    enabled = json.loads((directory / f"gdn-model-enabled{suffix}.json").read_text())
    baseline = json.loads(
        (directory / f"gdn-model-default-off{suffix}.json").read_text()
    )
    assert enabled.get("passed") and baseline.get("passed")
    for key in ("parameters", "model_files", "probe_sha256"):
        assert enabled[key] == baseline[key], f"comparison changed {key}"
    assert len(enabled["records"]) == len(baseline["records"]) == 4
    for a, b in zip(enabled["records"], baseline["records"]):
        assert a["label"] == b["label"]
        assert a["prompt_sha256"] == b["prompt_sha256"]
        assert a["token_ids"] == b["token_ids"], (a, b)
    if dflash:
        assert enabled["mixed"]["prompt_sha256"] == baseline["mixed"]["prompt_sha256"]
        assert enabled["mixed"]["token_ids"] == baseline["mixed"]["token_ids"]
    on_hit = enabled["records"][2]["cached_tokens"]
    off_hit = baseline["records"][2]["cached_tokens"]
    verdict = {
        "outputs_identical": True,
        "enabled_hit": on_hit,
        "off_hit": off_hit,
        "enabled_only_hit": on_hit > 0 and off_hit == 0,
        "cold_seconds_on_off": [
            enabled["records"][0]["elapsed_seconds"],
            baseline["records"][0]["elapsed_seconds"],
        ],
        "divergence_seconds_on_off": [
            enabled["records"][2]["elapsed_seconds"],
            baseline["records"][2]["elapsed_seconds"],
        ],
        "cold_peak_allocated_bytes_on_off": [
            max(rank["peak_allocated"] for rank in arm["records"][0]["memory_per_rank"])
            for arm in (enabled, baseline)
        ],
    }
    print(json.dumps(verdict))
    (directory / ("comparison" + suffix + ".json")).write_text(
        json.dumps(verdict, indent=2) + "\n"
    )
    if require_off_miss:
        assert on_hit > 0 and off_hit == 0, (
            "requested enabled-hit/default-off-miss criterion was not observed"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=("enabled", "default-off", "enabled-dflash", "default-off-dflash"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound")
    parser.add_argument("--drafter", default="/app/models/Qwen3.8-27B-DFlash2-W4A16")
    parser.add_argument(
        "--kv-cache-bytes",
        type=int,
        default=4_000_000_000,
        help="fixed KV pool per GPU, identical in both arms",
    )
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--material-tokens", type=int, default=2000)
    parser.add_argument("--shared-tokens", type=int, default=1200)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--require-multiple-export", action="store_true",
                        help="require more than one checkpoint exported in one prefill")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--require-off-miss", action="store_true")
    args = parser.parse_args()
    if args.kv_cache_bytes <= 0:
        parser.error("--kv-cache-bytes must be positive")
    if args.chunk_size <= 0 or args.tensor_parallel_size <= 0:
        parser.error("chunk size and tensor parallel size must be positive")
    if not 0 < args.shared_tokens < args.material_tokens <= 11000:
        parser.error("require 0 < shared tokens < material tokens <= 11000")
    if args.compare_only:
        compare_results(
            args.output_dir, args.variant.endswith("dflash"), args.require_off_miss
        )
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant = args.variant
    os.environ["VLLM_GDN_PREFILL_CHECKPOINTS"] = (
        "1" if variant.startswith("enabled") else "0"
    )
    from vllm import LLM, SamplingParams

    enabled = variant.startswith("enabled")
    speculative = variant.endswith("dflash")
    result = {"variant": variant, "records": []}
    extra = {}
    if speculative:
        extra["speculative_config"] = {
            "method": "dflash",
            "num_speculative_tokens": 7,
            "model": args.drafter,
        }
    parameters = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=11264,
        max_num_seqs=2,
        max_num_batched_tokens=args.chunk_size,
        gpu_memory_utilization=0.80,
        kv_cache_memory_bytes=args.kv_cache_bytes,
        kv_cache_dtype="auto",
        attention_backend="FLASH_ATTN",
        enforce_eager=True,
        enable_prefix_caching=True,
        language_model_only=True,
        mamba_cache_mode="align",
        prefix_cache_retention_interval=None,
        seed=42,
        worker_extension_cls="test_gdn_checkpoint_model.GDNProbe",
        **extra,
    )
    result["parameters"] = parameters
    result["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["model_files"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ([args.model, args.drafter] if speculative else [args.model])
        for name in ("config.json", "model.safetensors.index.json", "tokenizer.json")
        if (path := Path(directory) / name).is_file()
    }
    llm = LLM(**parameters)
    try:
        result["layers"] = llm.collective_rpc("checkpoint_probe_install")
        print("LAYER_SPECS", json.dumps(result["layers"]), flush=True)
        tokenizer = llm.get_tokenizer()
        material = "The unique archive passcode is 483729.\n" + "".join(
            f"Archive entry {i}: the blue cedar tree stands beside a quiet river.\n"
            for i in range(2000)
        )
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": material}],
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(tokens, Mapping):
            tokens = tokens["input_ids"]
        if tokens and isinstance(tokens[0], list):
            tokens = tokens[0]
        print("PROMPT_LENGTH", len(tokens), flush=True)
        prefix = tokens[:args.material_tokens]
        assert len(prefix) == args.material_tokens
        tail = tokenizer.encode(
            "\nWhat is the unique archive passcode? Reply with digits only."
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        params = SamplingParams(temperature=0, max_tokens=12)

        def generate(label, ids):
            llm.collective_rpc("checkpoint_probe_reset")
            begin = time.monotonic()
            out = llm.generate({"prompt_token_ids": ids}, params, use_tqdm=False)[0]
            elapsed = time.monotonic() - begin
            row = {
                "label": label,
                "prompt_tokens": len(ids),
                "prompt_sha256": hashlib.sha256(bytes(str(ids), "utf8")).hexdigest(),
                "cached_tokens": out.num_cached_tokens,
                "text": out.outputs[0].text,
                "elapsed_seconds": elapsed,
                "memory_per_rank": llm.collective_rpc("checkpoint_probe_memory"),
                "token_ids": out.outputs[0].token_ids,
                "checkpoints": llm.collective_rpc("checkpoint_probe_read"),
                "prefill_forwards": llm.collective_rpc("checkpoint_probe_forwards"),
                "draft_calls": llm.collective_rpc("checkpoint_probe_drafting"),
            }
            result["records"].append(row)
            print("RESULT", json.dumps(row), flush=True)
            return row

        # Unscored same-size warmup, then clear every prefix entry. The first
        # measured request must still be cold in both fresh-process arms.
        warmup = (
            [tokenizer.encode("Independent", add_special_tokens=False)[0]]
            + prefix[1:]
            + tail
        )
        llm.generate({"prompt_token_ids": warmup}, params, use_tqdm=False)
        llm.reset_prefix_cache()
        cold = generate("cold", prefix + tail)
        assert cold["cached_tokens"] == 0, cold
        # Fixed before the model runs, independently of exported offsets. The
        # default 1200/2000 case diverges inside the shipped 2048-token chunk;
        # the historical long case is --material-tokens 8000 --shared-tokens 4800.
        shared = args.shared_tokens
        generate(
            "interleave",
            tokenizer.encode("Count from 1 to 3.", add_special_tokens=False),
        )
        divergent_tail = tokenizer.encode(
            "\nThis is a different archive review. Ignore the remaining entries. "
            "What is the unique archive passcode? Reply with digits only."
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )
        divergent = prefix[:shared] + divergent_tail
        first_difference = next(
            i for i, (a, b) in enumerate(zip(prefix + tail, divergent)) if a != b
        )
        assert first_difference == shared, first_difference
        result["first_difference_token"] = first_difference
        resumed = generate("deep-divergence", divergent)
        repeated = generate("warm-repeat", prefix[:shared] + divergent_tail)
        result["shared_prefix_tokens"] = shared
        result["shared_fraction"] = shared / len(prefix)
        offset = resumed["cached_tokens"]
        assert "483729" in cold["text"], cold
        assert "483729" in resumed["text"], resumed
        assert resumed["token_ids"] == repeated["token_ids"]
        if enabled:
            assert 0 < resumed["cached_tokens"] < shared, resumed
            assert all(cold["checkpoints"]), "Every rank must actually export states"
            if args.require_multiple_export:
                assert all(
                    any(len(record["slices"]) > 1 for record in rank)
                    for rank in cold["checkpoints"]
                ), "Requested multiple states in one prefill were not observed"
        if speculative:
            assert all(cold["draft_calls"]), "DFlash propose must actually execute"
        if speculative:
            long_ids = (
                [tokenizer.encode("Independent mixed", add_special_tokens=False)[0]]
                + tokens[:11000]
                + tail
            )
            short_ids = tokenizer.encode("Count from 1 to 3.", add_special_tokens=False)
            llm.collective_rpc("checkpoint_probe_reset")
            outputs = llm.generate(
                [{"prompt_token_ids": long_ids}, {"prompt_token_ids": short_ids}],
                params,
                use_tqdm=False,
            )
            result["mixed"] = {
                "outputs": [o.outputs[0].text for o in outputs],
                "token_ids": [o.outputs[0].token_ids for o in outputs],
                "prompt_sha256": [
                    hashlib.sha256(str(ids).encode()).hexdigest()
                    for ids in (long_ids, short_ids)
                ],
                "checkpoints": llm.collective_rpc("checkpoint_probe_read"),
            }
            assert "483729" in outputs[0].outputs[0].text
        result["checkpoint_offset"] = offset
        result["passed"] = True
    finally:
        (args.output_dir / f"gdn-model-{variant}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
