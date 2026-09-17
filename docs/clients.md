# Clients

Pointing coding agents and OpenAI-protocol clients at the server.

[← back to the main README](../README.md)

## OpenCode (optional)

If you want to point [OpenCode](https://opencode.ai) at this local vLLM server,
add an `opencode.json` file in the directory where you run it from, or place it
in `~/.config/opencode/`. The base URL must include `/v1`, and the model name
must match what the server is serving — `http://127.0.0.1:18020/v1` and
`qwen3.8-27b` for the default single-user setup shown here.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "qwen-local/qwen3.8-27b",
  "provider": {
    "qwen-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen 3.8 27B (local RTX 3090)",
      "options": {
        "baseURL": "http://127.0.0.1:18020/v1",
        "apiKey": "$VLLM_API_KEY"
      },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen 3.8 27B",
          "limit": {
            "context": 65536,
            "output": 8192
          }
        }
      }
    }
  }
}
```

With the server running, install the OpenCode CLI and launch it from the same
location:

```bash
opencode
```

If auth is disabled, any placeholder value works for `apiKey`; if you enabled
`VLLM_API_KEY`, set the same value here. The `context` value above assumes the
default `CTX=fast` profile (`65536`); `CTX=long` is `131072`, and `CTX=huge` is
`245760`. Under-declaring the window can make the client silently truncate
context.

To check the numbers on your own card: `bash verify.sh` (also probes the live
server and prints which attention backend and KV pool it came up with), then
`bash bench/run_benchmarks.sh batch` or `... single` reproduces the tables
above against the running server (`--prefill` and `--long` add the prefill
matrix and the long-context rows), `bash bench/real_rep.sh <tag> 3 0` repeats
the single-stream row, and `python bench/quality_battery.py <tag>` the
perplexity / GSM8K rows. For the concurrency rows,
`python bench/conc_ladder.py --n 1,2,4,8 --ctx-tokens 4096`; for the prompt-length
bug, `python bench/residue_sweep.py <tag>` (all 128 residues) with
`python bench/verbatim.py` as its offline self-test.
