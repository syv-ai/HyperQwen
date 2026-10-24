"""Check the installed xgrammar carries mlc-ai/xgrammar#989 and the Qwen template-newline fix. CPU only, no model.

On Qwen3-Coder's tool format (the `qwen_3_coder` structural tag), a string parameter with maxLength 40:
- #989 makes the bounded value stop at the closing tag, so a value containing "</parameter>" is refused
  (the released 0.2.7 / 0.2.8 accept it);
- the fix keeps the template's two framing newlines out of the bound, so "\\n" + 40 chars + "\\n" is accepted
  (#989 alone accepts 38).
Usage: python xgrammar/check.py
"""
import importlib.metadata

import xgrammar as xg

tools = [{"type": "function", "function": {"name": "save", "parameters": {
    "type": "object", "properties": {"content": {"type": "string", "maxLength": 40}}, "required": ["content"]}}}]
vocab = xg.TokenizerInfo([bytes([i]) for i in range(256)])
grammar = xg.GrammarCompiler(vocab, cache_enabled=False).compile_structural_tag(
    xg.get_model_structural_tag("qwen_3_coder", tools=tools, tool_choice="auto", reasoning="disabled"))


def accepts(value):
    call = f"<tool_call>\n<function=save>\n<parameter=content>{value}</parameter>\n</function>\n</tool_call>"
    return xg.GrammarMatcher(grammar).accept_string(call)


cases = {
    "framed 40 chars (the fix)": ("\n" + "x" * 40 + "\n", True),
    "framed 41 chars": ("\n" + "x" * 41 + "\n", False),
    "closing tag inside the value (#989)": ("\na</parameter>b\n", False),
}
bad = [name for name, (value, want) in cases.items() if accepts(value) != want]
version = importlib.metadata.version("xgrammar")
if bad:
    raise SystemExit(f"xgrammar {version}: failed {bad}")
print(f"xgrammar {version}: #989 and the Qwen framing fix present ({len(cases)} cases)")
