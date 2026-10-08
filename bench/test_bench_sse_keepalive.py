"""`vllm bench serve` against a server that sends an SSE keep-alive before the first token (patches/bench-sse-keepalive.patch).

Stock 0.30 fails the request with "Never received a valid chunk to calculate TTFT"; patched, it returns the text.
No GPU, no model: a local aiohttp server on port 18999 plays the vLLM side.
Exits 1 if any request fails, so a run that compares a stock file with a patched one exits 1.

  CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_bench_sse_keepalive.py              # the installed vLLM
  CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_bench_sse_keepalive.py a=/path/x.py  # compare files
"""
import asyncio, importlib.util, sys
from aiohttp import web
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
async def sse(request):
    r = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await r.prepare(request)
    await r.write(b": keep-alive\n\n"); await asyncio.sleep(0.05)
    for t in ("a", "b", "c"):
        await r.write(b'data: {"choices":[{"text":"%s"}]}\n\n' % t.encode()); await asyncio.sleep(0.02)
    await r.write(b'data: {"choices":[],"usage":{"completion_tokens":3,"prompt_tokens":5}}\n\n'); await r.write(b"data: [DONE]\n\n")
    return r
async def main(paths):
    app = web.Application(); app.router.add_post("/v1/completions", sse)
    runner = web.AppRunner(app); await runner.setup(); await web.TCPSite(runner, "127.0.0.1", 18999).start()
    import aiohttp
    ok = True
    for label, path in paths:
        m = load(path, "erf_" + label)
        inp = m.RequestFuncInput(prompt="x", api_url="http://127.0.0.1:18999/v1/completions", prompt_len=5, output_len=3, model="m")
        async with aiohttp.ClientSession() as s:
            out = await m.async_request_openai_completions(inp, session=s)
        print(f"{label:8s} success={out.success} text={out.generated_text!r} error={out.error[:60]!r}")
        ok = ok and out.success
    await runner.cleanup()
    return ok
if len(sys.argv) > 1:
    ok = asyncio.run(main([a.split("=", 1) for a in sys.argv[1:]]))
else:
    import vllm.benchmarks.lib.endpoint_request_func as erf
    ok = asyncio.run(main([("installed", erf.__file__)]))
sys.exit(0 if ok else 1)
