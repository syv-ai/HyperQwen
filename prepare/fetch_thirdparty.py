"""Fetch a third-party Qwen3.8-27B checkpoint in this repo's served shape and
say what to run next. The default is the worked example this script shipped
with (PR #37): philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ, an AWQ
quant of the de-refused Qwen3.8-27B with the vision tower and the grafted MTP
head preserved.

  venv/bin/python prepare/fetch_thirdparty.py [hf-repo] [dst_dir]
  # default: philbert440/... -> models/Qwen3.8-27B-Uncensored-W4A16

~18.6 GB for the default. A checkpoint like it is NOT servable on 24 GB as it
ships, for the same reason the base model is not (bf16 lm_head, bf16
embeddings, bf16 MTP module) -- and the three prepare/quant_*.py scripts
read a shard into RAM whole, so they cannot rewrite its one 18.6 GB shard;
prepare/quant_heads_stream.py streams it. This script prints the exact
commands when the download finishes.
(A ready-made alternative needing none of this is listed in the README's
"Third-party checkpoints" section.)
"""
import os, sys
from huggingface_hub import snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
DEFAULT_REPO = "philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
REPO = args[0] if args else DEFAULT_REPO
default_dst = ("Qwen3.8-27B-Uncensored-W4A16" if REPO == DEFAULT_REPO
               else REPO.split("/")[-1])
D = (args[1] if len(args) > 1 else os.path.join(ROOT, "models", default_dst)).rstrip("/")
os.makedirs(D, exist_ok=True)

# chat_template.jinja is not *.json and the server needs it: without it vLLM falls back
# to the tokenizer's built-in template, which is not the one these checkpoints were
# tuned with (and does not emit the XML tool-call format --tool-call-parser
# qwen3_coder reads). *.txt covers merges.txt-style tokenizers that ship no
# tokenizer.json.
snapshot_download(REPO, local_dir=D,
                  allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors",
                                  "README.md"])

rel = os.path.relpath(D, ROOT)
print(f"\ncheckpoint downloaded: {rel}")
print("it is not servable yet -- requantize the heads (CPU only, ~15 min, needs ~20 GB free disk\n"
      "for the .bak-orig originals):\n"
      f"  venv/bin/python prepare/quant_heads_stream.py {rel}\n"
      f"  venv/bin/python prepare/build_draft_vocab.py {rel} --ids prepare/draft_vocab_ids.json\n"
      f"then:  MODEL=$PWD/{rel} bash single-user/start_qwen.sh")
