# SPDX-License-Identifier: Apache-2.0
"""CPU HTTP regressions for the pinned benchmark's actual probe functions."""
import ast
import asyncio
import contextlib
from dataclasses import dataclass, replace
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import aiohttp
from aiohttp import web

METRICS = """vllm:spec_decode_num_drafts_total 2
vllm:spec_decode_num_draft_tokens_total 4
vllm:spec_decode_num_accepted_tokens_total 3
vllm:diffusion_num_denoising_steps_total 2
vllm:diffusion_num_canvas_positions_total 4
vllm:diffusion_num_committed_tokens_total 3
"""


@dataclass
class Request:
    prompt: str = "original"
    prompt_len: int = 2


class BenchmarkHeaderTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        selected = os.environ.get("VLLM_SOURCE")
        root = Path(selected) if selected else Path(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))
        source = root/"benchmarks/serve.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"_benchmark_probe_headers", "_align_prompts_to_server_tokenizer", "fetch_spec_decode_metrics",
                 "fetch_diffusion_metrics", "SpecDecodeMetrics", "DiffusionMetrics"}
        definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in names]
        cls.module = ModuleType("probe_headers_fixture")
        sys.modules[cls.module.__name__] = cls.module
        cls.addClassCleanup(sys.modules.pop, cls.module.__name__, None)
        cls.module.__dict__.update(aiohttp=aiohttp, asyncio=asyncio, contextlib=contextlib, dataclass=dataclass,
                                  replace=replace, os=os, SampleRequest=Request)
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *definitions], type_ignores=[])), str(source), "exec"), cls.module.__dict__)

    async def asyncSetUp(self):
        self.seen = []
        self.expected = "Bearer fixture-explicit"
        self.status = 200

        async def respond(request):
            headers = request.headers.getall("Authorization", [])
            self.seen.append((request.path, headers, request.headers.get("X-Trace")))
            if headers != ([] if self.expected is None else [self.expected]):
                return web.Response(status=401)
            if self.status != 200:
                return web.Response(status=self.status)
            if request.path == "/metrics":
                return web.Response(text=METRICS)
            if request.path == "/tokenize":
                return web.json_response({"tokens": [1, 2, 3, 4]})
            return web.json_response({"prompt": "corrected"})

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", respond)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def exercise(self, headers):
        original = dict(headers) if headers is not None else None
        self.seen.clear()
        async with aiohttp.ClientSession() as session:
            spec = await self.module.fetch_spec_decode_metrics(self.url, session, headers=headers)
            diffusion = await self.module.fetch_diffusion_metrics(self.url, session, headers=headers)
        aligned = await self.module._align_prompts_to_server_tokenizer(self.url, "fixture", [Request()], headers=headers)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(diffusion)
        self.assertEqual(aligned[0].prompt, "corrected")
        self.assertEqual(headers, original, "caller headers must not be mutated")
        self.assertEqual(len(self.seen), 5)
        return self.seen

    async def test_explicit_header_wins_for_every_http_casing(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-environment"}):
            for key in ("Authorization", "authorization", "aUtHoRiZaTiOn"):
                with self.subTest(header=key):
                    seen = await self.exercise({key: "Bearer fixture-explicit", "X-Trace": "trace"})
                    self.assertTrue(all(trace == "trace" for _, _, trace in seen))

    async def test_environment_key_fills_absent_authorization(self):
        self.expected = "Bearer fixture-environment"
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fixture-environment"}):
            await self.exercise({"X-Trace": "trace"})
            await self.exercise(None)

    async def test_explicit_key_works_without_environment_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            await self.exercise({"authorization": "Bearer fixture-explicit"})

    async def test_unkeyed_server_still_works(self):
        self.expected = None
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            await self.exercise(None)

    async def test_probe_failures_keep_existing_return_contract(self):
        self.expected = None
        self.status = 401
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            async with aiohttp.ClientSession() as session:
                self.assertIsNone(await self.module.fetch_spec_decode_metrics(self.url, session))
                self.assertIsNone(await self.module.fetch_diffusion_metrics(self.url, session))
            requests = [Request()]
            self.assertIs(await self.module._align_prompts_to_server_tokenizer(self.url, "fixture", requests), requests)


if __name__ == "__main__":
    unittest.main()
