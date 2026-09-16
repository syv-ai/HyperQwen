# Quality

Does the quantization stack cost accuracy? IFBench, perplexity and GSM8K against this exact serving setup, per configuration.

[← back to the main README](../README.md)

Three checks against this exact serving stack:

**IFBench** ([AllenAI's](https://github.com/allenai/IFBench) out-of-distribution
instruction-following benchmark, 299 prompts, official eval scripts), thinking
enabled at `reasoning_effort: xhigh` (the **eval protocol**, matching the model
card default — not the house serving default, which is `medium`; see
[house-serving.md](house-serving.md)), model-default sampling:

| accuracy | prompt-level | instruction-level |
|---|---|---|
| strict, W4A16 stack | **78.3** | 79.9 |
| loose, W4A16 stack | 81.7 | 82.8 |
| strict, batch mode default (int8 MLP activations, fp16 state) | **78.3** | 80.5 |
| loose, batch mode default | 80.3 | 82.3 |

Qwen's [model card](https://huggingface.co/Qwen/Qwen3.8-27B) reports **79.5**
for the unquantized model, so the W4A16 quantization stack costs about one
point on the headline metric (prompt-level strict), and the batch-mode int8
activations cost nothing measurable on it (the two runs trade places within
sampling noise on the sub-metrics).

**Perplexity** on ~33k tokens of held-out text (English Wikipedia, Danish web
text, Python source), and **GSM8K** (200 test questions, greedy, thinking off):

| batch-mode config | PPL en | PPL da | PPL code | PPL all | GSM8K | 64-conc e2e (128/512) |
|---|---|---|---|---|---|---|
| W4A16, fp32 state (as shipped before) | 10.68 | 10.85 | 3.05 | 8.045 | — | 516 tok/s |
| W4A16, fp16 state | 10.68 | 10.85 | 3.05 | 8.044 | 95.5% | 707 tok/s |
| + int8 activations, gate/up projections | 10.74 | 10.93 | 3.10 | 8.12 (+0.9%) | 95.5% | 787 tok/s |
| + int8 activations, whole MLP (**default**) | 10.88 | 11.05 | 3.15 | 8.22 (+2.2%) | 95.0% | 942 tok/s |
| + int8 activations, all linear layers | 10.93 | 11.29 | 3.20 | 8.34 (+3.7%) | — | 1,042 tok/s* |

Reading: the 16-bit recurrent state is free; every int8-activation step costs a
little perplexity, mostly on code, and buys throughput. The default takes the
middle row; `INT8_LAYERS=gate_up` and `INT8_ACT=` (off) are one env var away,
and `INT8_LAYERS=.` gives you the last row.

*The all-layers row needs `GPU_UTIL=0.95`: quantizing the activations of every
linear (not just the MLP) adds enough transient scratch that batch mode's 0.972
runs out of memory inside the GDN chunk kernel once ~17 requests are resident.
The throughput columns were re-measured on the current stack (two passes each);
the perplexity columns date from when those rows were first measured.

Single-user mode is W4A16 (int8 activations buy nothing at batch size 1) and
speculative decoding is exact by construction, so with the base requantization
its quality is the W4A16 row. The single-user **fast variant** additionally
runs lm_head at int4 (GPTQ, calibrated on the model's own hidden states):

| single-user config | PPL en | PPL da | PPL code | PPL all | GSM8K | C1 tok/s (default / greedy) |
|---|---|---|---|---|---|---|
| base requantization (int8 lm_head), new draft vocab | 10.68 | 10.85 | 3.05 | 8.045 | 95.5% | 107 / 109 |
| int4 lm_head, round-to-nearest (not shipped) | 10.81 | 11.09 | 3.07 | 8.17 (+1.5%) | — | 109 / 112 |
| **fast variant**: int4 lm_head GPTQ + int4 MTP GPTQ | 10.77 | 10.91 | 3.06 | 8.095 (+0.6%) | 96.5% | ~114 / ~124 |

The MTP module's precision never touches output quality (drafts are verified
exactly); it only moves acceptance, and the calibrated int4 keeps it.
