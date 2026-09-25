"""Assemble models/Qwen3.8-27B-W4A16-AutoRound-fast: the single-user "fast" variant
(int4-GPTQ lm_head + MTP module, own-output draft vocabulary) from the base model dir
plus the prebuilt tensors on the Hub (syvai/qwen3.8-27b-3090-fast-variant, ~1 GB).

Shards 1-6, tokenizer and template files are hardlinked from the base dir (no extra
disk); shard 7, model_extra_tensors, config, index and the draft vocab ids are
downloaded. To rebuild the tensors yourself instead, see drafter/README.md.

Every copy goes through a temp file and a rename (prepare/atomic_publish.py), and the
index is copied last: docker/prepare.sh's state() treats the fast variant as present once
its index exists, so a killed run must never leave a partial file behind it (#195).

  venv/bin/python prepare/fetch_fast_variant.py [base_dir] [dst_dir]
"""
import os, sys, shutil
from huggingface_hub import snapshot_download

from atomic_publish import publish


def copy_atomic(src, dst):
    shutil.copyfile(src, dst + ".tmp")
    publish(dst + ".tmp", dst)


HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
S = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound")).rstrip("/")
D = (sys.argv[2] if len(sys.argv) > 2 else S + "-fast").rstrip("/")
S = os.path.realpath(S)
assert os.path.exists(os.path.join(S, "config.json")), f"base model not found at {S} (README: Setup)"
os.makedirs(D, exist_ok=True)
for f in sorted(os.listdir(S)):
    if f.startswith("model-0000") and f.endswith(".safetensors") and f != "model-00007-of-00007.safetensors":
        if not os.path.exists(os.path.join(D, f)):
            os.link(os.path.join(S, f), os.path.join(D, f))
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
          "processor_config.json", "quantization_config.json"]:
    if os.path.exists(os.path.join(S, f)) and not os.path.exists(os.path.join(D, f)):
        copy_atomic(os.path.join(S, f), os.path.join(D, f))
# Stop here rather than assemble a dir that cannot be served: transformers treats a
# missing tokenizer as an empty vocabulary instead of an error, and vLLM only notices
# much later, as "ReasoningConfig: failed to tokenize reasoning strings".
assert os.path.exists(os.path.join(D, "tokenizer.json")), (
    f"no tokenizer.json in {S} — the base model dir is incomplete; re-run the download "
    f"(README: Setup) before building the fast variant")
hub = snapshot_download("syvai/qwen3.8-27b-3090-fast-variant",
                        allow_patterns=["model-00007-of-00007.safetensors", "model_extra_tensors.safetensors",
                                        "mtp_draft_vocab_ids.pt", "config.json", "model.safetensors.index.json"])
# model.safetensors.index.json stays last in this list: it is the file state() checks.
for f in ["model-00007-of-00007.safetensors", "model_extra_tensors.safetensors", "mtp_draft_vocab_ids.pt",
          "config.json", "model.safetensors.index.json"]:
    copy_atomic(os.path.realpath(os.path.join(hub, f)), os.path.join(D, f))
print("fast variant ready:", D)
print("serve with: CTX=fast bash single-user/start_qwen.sh   (picked up automatically when the dir exists)")
