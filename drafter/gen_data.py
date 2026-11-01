"""Generate self-distillation data: run the served model offline over data/prompts.jsonl
(model's default sampling params, thinking on/off per prompt) and store token ids.
Output: data/gen.jsonl lines {"id","src","think","prompt_ids","output_ids","finish"}.
Resumable: skips ids already present in the output file.

  VLLM_MARLIN_INPUT_DTYPE=int8 VLLM_MARLIN_INT8_INCLUDE_RE=mlp python gen_data.py [--limit N] [--chunk 512]
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from generation_state import load_completed, exclusive_output, OutputLockedError

MODEL = os.environ.get("MODEL", os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound"))
D = os.path.join(HERE, "data")
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
CHUNK = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else 512
MAX_THINK = int(os.environ.get("MAX_THINK", 2048))
MAX_NOTHINK = int(os.environ.get("MAX_NOTHINK", 1024))


def main():
    out_path = f"{D}/gen.jsonl"
    try:
        with exclusive_output(out_path):
            generate(out_path)
    except OutputLockedError as exc:
        sys.exit(str(exc))


def generate(out_path):
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [json.loads(l) for l in open(f"{D}/prompts.jsonl")]
    if LIMIT:
        prompts = prompts[:LIMIT]
    done = load_completed(out_path)
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{len(prompts)} prompts, {len(done)} done, {len(todo)} to go", flush=True)

    llm = LLM(
        model=MODEL,
        served_model_name="qwen3.8-27b",
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.93)),
        max_model_len=8192,
        max_num_seqs=int(os.environ.get("MAX_SEQS", 64)),
        max_num_batched_tokens=2048,
        kv_cache_dtype="fp8",
        mamba_ssm_cache_dtype="float16",
        language_model_only=True,
        enable_prefix_caching=False,
        compilation_config={"max_cudagraph_capture_size": 64, "custom_ops": ["+rms_norm", "+silu_and_mul"]},
    )
    base = llm.get_default_sampling_params()
    print("default sampling params:", base, flush=True)


    def sp(think):
        p = base.clone()
        p.max_tokens = MAX_THINK if think else MAX_NOTHINK
        p.n = 1
        return p


    t0 = time.time()
    ntok = 0
    with open(out_path, "a") as f:
        for i in range(0, len(todo), CHUNK):
            batch = todo[i:i + CHUNK]
            inputs, params, meta = [], [], []
            for p in batch:
                try:
                    text = tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=p["think"])
                except Exception as e:
                    print("template fail", p["id"], e)
                    continue
                ids = tok(text, add_special_tokens=False)["input_ids"]
                if len(ids) > 3000:
                    continue
                inputs.append({"prompt_token_ids": ids})
                params.append(sp(p["think"]))
                meta.append(p)
            outs = llm.generate(inputs, params, use_tqdm=False)
            for p, o in zip(meta, outs):
                c = o.outputs[0]
                rec = {"id": p["id"], "src": p["src"], "think": p["think"],
                       "prompt_ids": list(o.prompt_token_ids), "output_ids": list(c.token_ids),
                       "finish": c.finish_reason}
                f.write(json.dumps(rec) + "\n")
                ntok += len(c.token_ids)
            f.flush()
            el = time.time() - t0
            print(f"[{i + len(batch)}/{len(todo)}] {ntok} output tokens, {ntok / el:.0f} tok/s, {el / 60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()
