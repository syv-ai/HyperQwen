# House serving defaults

The weights file stays put. Which stack serves it is the variable. This
repo is the patched vLLM W4A16 path (OpenAI-compatible, default port
18020). llama.cpp / Ollama / GGUF is a different stack, not this one.

These defaults apply to a live house listener **without a restart**. They
do not retune `CTX=huge`, 262K reservations, or `reasoning_effort=xhigh`.

| Knob | House | Not house |
|---|---|---|
| Window | `CTX=fast`, `max_model_len` **65536** | `CTX=long` (~150k), `CTX=huge` / 262K lab profiles |
| Client thinking | `chat_template_kwargs.reasoning_effort=medium` | `xhigh` (model-card default; IFBench protocol in [quality.md](quality.md)) |
| Listener | this image on **:18020** | a second vLLM on :8001, or llama.cpp on :11434 |

Pin the window in `.env` with `MAX_LEN=65536`. Clients should send medium
reasoning unless they are reproducing the IFBench row.

Do not restart a healthy `single` container to “apply” this doc.
