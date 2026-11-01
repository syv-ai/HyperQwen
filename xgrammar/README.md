# xgrammar/: a source build of xgrammar for bounded strings in Qwen tool calls

With `TOOL_STRICT=parameter`, every Qwen3 tool call is constrained by xgrammar's `qwen_3_coder` structural tag, whose
string parameters are raw text between `<parameter=name>` and `</parameter>`. On the released xgrammar (0.2.7, which
vLLM 0.31 pins, and 0.2.8, which 0.32 pins), any `maxLength` on such a string puts that text on a slow path: ~78 ms
per token mask against ~0.002 ms without the bound, on the Qwen3.8 tokenizer. Served on an RTX 4090 (vLLM 0.31,
DFlash2, a ~10k-token prompt, one file-writing call with `"maxLength": 200000` on its content), decode fell from
~170 to 7.3 tok/s with the GPU idle. syv-ai/HyperQwen#379 has the measurements.

`install.sh` builds xgrammar at the commit in `PIN` and installs it over the pinned release:

- mlc-ai/xgrammar#989 (by anton-iermolenko, open): extends #955's delimiter-aware bounded body from GLM's XML format to
  every XML format, Qwen's included. This is the speed fix.
- One commit on top, anton-iermolenko/xgrammar#1: Qwen's chat template writes `<parameter=k>\n` + value +
  `\n</parameter>` and vLLM's Qwen3 parser strips one newline on each side; #989 counts both toward the bound, so a
  framed value got `maxLength - 2` characters. The commit keeps those two newlines outside the bound.

Measured with both (vLLM 0.31, the same request): 158.9 tok/s at `maxLength` 200000, GPU 99%, and tool-call
arguments byte-identical to 0.2.7's. On this image (vLLM 0.32, an RTX 4090, the same request, one run per row):
181.2, 180.9 and 138.2 tok/s at `maxLength` 200000, 20000 and 2000, against 5.9, 6.9 and 7.4 on xgrammar 0.2.8, and
the same tool-call arguments on both builds.

`check.py` runs at the end of the install, CPU only: it fails on 0.2.7 and 0.2.8 (a value containing `</parameter>`
is accepted) and on #989 alone (a framed 40-character value under `maxLength` 40 is refused).

Drop this directory, its Dockerfile step and its install line when vLLM pins an xgrammar release that carries #989
and the framing fix.
