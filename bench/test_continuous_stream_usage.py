"""CPU regressions for continuous usage during Qwen tool-argument buffering.

Run after patches/apply.sh with the pinned vLLM runtime:
  CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_continuous_stream_usage.py
The real parser and streaming generator consume synthetic engine outputs;
no model, server or GPU is needed.
"""

import asyncio
import json
import re
import unittest
from types import SimpleNamespace

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.parser.qwen3 import Qwen3Parser


class CharacterTokenizer:
    all_special_tokens = []
    all_special_ids = []

    def get_vocab(self):
        return {
            **{chr(i): i for i in range(128)},
            "<think>": 1000,
            "</think>": 1001,
            "<tool_call>": 1002,
            "</tool_call>": 1003,
        }

    def decode(self, tokens, **kwargs):
        vocabulary = {value: key for key, value in self.get_vocab().items()}
        return "".join(vocabulary[token] for token in tokens)


async def serve(continuous=True, include_usage=True, include_reasoning=True):
    arguments = {"commands": ["ordinary sample sentence " * 100]}
    raw = (
        "Working through the sample.</think>\nChecking the result.\n"
        "<tool_call>\n<function=probe>\n<parameter=commands>"
        + json.dumps(arguments["commands"])
        + "</parameter>\n</function>\n</tool_call>"
    )
    pieces = re.findall(r"</?think>|</?tool_call>|.", raw, re.DOTALL)
    vocabulary = CharacterTokenizer().get_vocab()
    tool = {
        "type": "function",
        "function": {
            "name": "probe",
            "parameters": {
                "type": "object",
                "properties": {
                    "commands": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["commands"],
                "additionalProperties": False,
            },
        },
    }
    request = ChatCompletionRequest(
        model="offline",
        messages=[{"role": "user", "content": "offline"}],
        tools=[tool],
        stream=True,
        tool_choice="auto",
        include_reasoning=include_reasoning,
        stream_options={
            "include_usage": include_usage,
            "continuous_usage_stats": continuous,
        },
    )
    parser = Qwen3Parser(CharacterTokenizer(), request.tools)

    async def outputs():
        for index, character in enumerate(pieces):
            yield RequestOutput(
                request_id="offline",
                prompt=None,
                prompt_token_ids=[],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text=character,
                        token_ids=[vocabulary[character]],
                        cumulative_logprob=None,
                        logprobs=None,
                        finish_reason="stop" if index + 1 == len(pieces) else None,
                    )
                ],
                finished=index + 1 == len(pieces),
            )

    def unexpected(error):
        raise AssertionError(str(error)) from error

    server = SimpleNamespace(
        parser_cls=Qwen3Parser,
        _make_parser=lambda *args: parser,
        get_chat_request_role=lambda req: "assistant",
        enable_force_include_usage=False,
        enable_prompt_tokens_details=True,
        _include_reasoning_tokens_details=False,
        enable_log_outputs=False,
        enable_per_request_metrics=False,
        system_fingerprint=None,
        _raise_if_error=lambda *args: None,
        create_streaming_error_response=unexpected,
    )
    events = []
    async for frame in OpenAIServingChat.chat_completion_stream_generator(
        server,
        request,
        outputs(),
        "offline",
        "offline",
        [],
        CharacterTokenizer(),
        SimpleNamespace(final_usage_info=None),
    ):
        payload = frame.removeprefix("data: ").strip()
        if payload != "[DONE]":
            events.append(json.loads(payload))
    return events, arguments, len(pieces)


class StreamingUsageTests(unittest.TestCase):
    def check_output(self, events, arguments, hidden=False):
        chunks = [c for e in events for c in e.get("choices", [])]
        values = [
            d.get("function", {}).get("arguments") or ""
            for c in chunks
            for d in c.get("delta", {}).get("tool_calls") or []
        ]
        self.assertTrue(values, repr(events[-3:])[:1200])
        self.assertEqual(json.loads("".join(values)), arguments)
        self.assertEqual(sum(c.get("finish_reason") is not None for c in chunks), 1)
        reasoning = "".join(c.get("delta", {}).get("reasoning") or "" for c in chunks)
        self.assertEqual(reasoning, "" if hidden else "Working through the sample.")
        self.assertEqual(
            "".join(c.get("delta", {}).get("content") or "" for c in chunks),
            "\nChecking the result.\n",
        )
        self.assertTrue(
            all(
                c.get("token_ids") is None and c.get("logprobs") is None for c in chunks
            )
        )

    def test_continuous_usage_advances_during_buffered_array(self):
        events, arguments, total = asyncio.run(serve())
        self.check_output(events, arguments)
        counts = [e["usage"]["completion_tokens"] for e in events if e.get("usage")]
        self.assertEqual(counts, [*range(total + 1), total])
        empty = [
            e for e in events if e.get("choices") and e["choices"][0].get("delta") == {}
        ]
        self.assertGreater(len(empty), 1000)

    def test_hidden_reasoning_keeps_telemetry_without_leaking_text_or_ids(self):
        events, arguments, total = asyncio.run(serve(include_reasoning=False))
        self.check_output(events, arguments, hidden=True)
        self.assertEqual(
            [e["usage"]["completion_tokens"] for e in events if e.get("usage")],
            [*range(total + 1), total],
        )

    def test_final_usage_only_contract_is_preserved(self):
        events, arguments, total = asyncio.run(serve(continuous=False))
        self.check_output(events, arguments)
        self.assertEqual(
            [e["usage"]["completion_tokens"] for e in events if e.get("usage")], [total]
        )
        self.assertLess(len(events), total // 2)

    def test_no_usage_contract_is_preserved(self):
        events, arguments, total = asyncio.run(
            serve(continuous=True, include_usage=False)
        )
        self.check_output(events, arguments)
        self.assertFalse(any(e.get("usage") for e in events))
        self.assertLess(len(events), total // 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
