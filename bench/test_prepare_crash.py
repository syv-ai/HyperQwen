"""Crash-injection check for the model preparation scripts (#195).

docker/prepare.sh runs before every container start, under `set -e`, so a file that a
killed run leaves half-written can stop every later start. This builds a small synthetic
model (the base checkpoint's layout, plus a single-shard export) and runs prepare.sh's
own sequence on it -- state() executed from docker/prepare.sh, each pending step, the
chat-template writers -- and quant_heads_stream.py. It kills the process after each file
write, once with the write complete and once with the file cut to its first half. After
each kill one clean pass must complete, and every file, backups included, must match a
run without a kill. That run must also leave each backup equal to the file as it was
before its step.

  venv/bin/python bench/test_prepare_crash.py [step ...]   # CPU and Linux only, minutes

Steps: lm_head embed mtp draft fast harden translate stream (default: all). The kill
goes into the named steps only. Exit 0 when every case passes.
"""
import builtins
import hashlib
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
KILLED = 137
STEPS = ("lm_head", "embed", "mtp", "draft", "fast", "harden", "translate", "stream")
BACKUPS = (".bak", ".bak_embed", ".bak-mtp", ".bak-quant", ".bak-draft", ".bak-orig")
V, K = 512, 256  # vocab x hidden, multiples of the 128-wide quantization group
MTP = {"mtp.fc": (K, 2 * K), "mtp.layers.0.mlp.down_proj": (K, 2 * K),
       "mtp.layers.0.mlp.gate_proj": (2 * K, K), "mtp.layers.0.mlp.up_proj": (2 * K, K),
       **{f"mtp.layers.0.self_attn.{p}_proj": (K, K) for p in "qkvo"}}
HUB = ("model-00007-of-00007.safetensors", "model_extra_tensors.safetensors",
       "mtp_draft_vocab_ids.pt", "config.json", "model.safetensors.index.json")


class Stop(Exception):
    pass


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def build(root, prep, ids):
    """root/base: embed in shard 1, lm_head in shard 7, the bf16 MTP module in the extras
    shard, a tokenizer, and a chat template with both rewritten blocks in its first half.
    root/hub: the fast variant's Hub files. root/single: a single-shard AWQ-like export."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models

    g = torch.Generator().manual_seed(0)
    rnd = lambda *s: (torch.randn(*s, generator=g) * 0.02).to(torch.bfloat16)
    t = {"model.language_model.embed_tokens.weight": rnd(V, K), "model.language_model.norm.weight": rnd(K),
         "lm_head.weight": rnd(V, K), **{m + ".weight": rnd(*s) for m, s in MTP.items()}}

    def model(d, shards, sym):
        os.makedirs(d)
        for name, ts in shards.items():
            save_file(ts, f"{d}/{name}", metadata={"format": "pt"})
        dump({"metadata": {}, "weight_map": {k: n for n, ts in shards.items() for k in ts}},
             f"{d}/model.safetensors.index.json")
        w = {"num_bits": 4, "type": "int", "symmetric": sym, "group_size": 128, "strategy": "group"}
        if not sym:
            w["zp_dtype"] = "torch.int8"
        dump({"quantization_config": {"config_groups": {"group_0": {"targets": ["Linear"], "weights": w}},
                                      "ignore": ["lm_head", *MTP]}}, f"{d}/config.json")

    base = f"{root}/base"
    model(base, {"model-00001-of-00007.safetensors": {k: v for k, v in t.items() if k.startswith("model.")},
                 HUB[0]: {"lm_head.weight": t["lm_head.weight"]},
                 HUB[1]: {k: v for k, v in t.items() if k.startswith("mtp.")}}, True)
    model(f"{root}/single", {"model.safetensors": t}, False)
    Tokenizer(models.WordLevel({f"t{i}": i for i in range(V)}, unk_token="t0")).save(f"{base}/tokenizer.json")
    dump({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "t0"}, f"{base}/tokenizer_config.json")
    dump(list(range(0, V, 3)), ids)
    sys.path.insert(0, prep)
    tr = runpy.run_path(f"{prep}/translate_chat_template.py")
    old = runpy.run_path(f"{prep}/harden_chat_template.py")["OLD"]
    effort = [f"    {{%- {tr['ANCHOR']}'xhigh') %}}",
              f"    {{%- {tr['SHAPES'][0]} ['low', 'medium', 'xhigh'] %}}",
              f"        {{{{- {tr['SHAPES'][1]}'unknown reasoning effort') }}}}",
              f"    {{%- {tr['SHAPES'][2]} %}}"]
    with open(f"{base}/chat_template.jinja", "w") as f:
        f.write("\n".join(effort + [old] + [f"{{#- line {i} -#}}" for i in range(120)]) + "\n")

    hub = f"{root}/hub"
    os.makedirs(hub)
    save_file({"lm_head.weight_packed": torch.zeros(V, K // 8, dtype=torch.int32)}, f"{hub}/{HUB[0]}")
    save_file({"mtp.fc.weight_packed": torch.ones(K, K // 4, dtype=torch.int32)}, f"{hub}/{HUB[1]}")
    torch.save(torch.arange(0, V, 5), f"{hub}/{HUB[2]}")
    dump({"fast_variant": True}, f"{hub}/{HUB[3]}")
    dump({"weight_map": {"lm_head.weight_packed": HUB[0], "mtp.fc.weight_packed": HUB[1]}}, f"{hub}/{HUB[4]}")


# ---- kill injection: runs inside the forked step process only ----

_open = builtins.open


class Killer:
    """Counts file mutations under `root` and dies at mutation `at`. `torn` first cuts a
    write over a live file (not a *.tmp sibling) to half its size."""

    def __init__(self, root, at, torn):
        self.root, self.at, self.torn = os.path.realpath(root) + os.sep, at, torn
        self.n = self.quiet = 0

    def mine(self, p):
        return isinstance(p, (str, os.PathLike)) and os.path.realpath(p).startswith(self.root)

    def hit(self, p, write):
        if self.quiet or not self.mine(p):
            return
        self.n += 1
        if self.n == self.at:
            if self.torn and write and not os.fspath(p).endswith(".tmp"):
                with _open(p, "r+b") as f:
                    f.truncate(os.path.getsize(p) // 2)
            os._exit(KILLED)


class Tracked:
    """A writable file whose close() counts as one mutation."""

    def __init__(self, f, path, killer):
        self._f, self._path, self._killer = f, path, killer

    def __getattr__(self, name):
        attr = getattr(self._f, name)
        if not callable(attr):
            return attr
        return lambda *a, _keep=self, **kw: attr(*a, **kw)  # open(p).write(b): stay open

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if not self._f.closed:
            self._f.close()
            self._killer.hit(self._path, True)

    __del__ = close


def install(killer):
    import safetensors.torch
    import torch

    def after(fn, arg, write):
        def run(*a, **kw):
            killer.quiet += write  # a writer that opens its own file counts once
            try:
                r = fn(*a, **kw)
            finally:
                killer.quiet -= write
            killer.hit(a[arg], write)
            return r
        return run

    for mod, name, arg, write in ((os, "replace", 1, 0), (os, "rename", 1, 0), (os, "link", 1, 0),
                                  (os, "remove", 0, 0), (os, "unlink", 0, 0),
                                  (safetensors.torch, "save_file", 1, 1), (torch, "save", 1, 1)):
        setattr(mod, name, after(getattr(mod, name), arg, write))

    def tracked(file, mode="r", *a, **kw):
        f = _open(file, mode, *a, **kw)
        return Tracked(f, file, killer) if set(mode) & set("wax+") and killer.mine(file) else f

    builtins.open = io.open = tracked


def fork(fn, log):
    """Run fn() in a child with its output in `log`; return (exit code, what fn returned).
    Every tensor op runs in a child: libgomp does not survive a fork after it started."""
    sys.stdout.flush()
    sys.stderr.flush()
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.close(r)
            fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            try:
                os.write(w, str(fn()).encode())
                code = 0
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else int(e.code is not None)
                if not isinstance(e.code, (int, type(None))):
                    print(e.code, file=sys.stderr)
            except BaseException:
                traceback.print_exc()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    os.close(w)
    out = os.read(r, 64).decode()
    os.close(r)
    return os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]), out


# ---- the passes ----

def stepper(ctx, root, inject=(), at=None, torn=False, check=False):
    """A step runner for one pass. Returns False from a step when the kill fired."""
    seen = {}

    def step(name, script, args):
        killer = Killer(root, None if at is None else at - sum(seen.values()), torn) if name in inject else None

        def main():
            sys.argv, sys.path[:0] = [script, *args], [ctx.prep]
            if name == "fast":
                import huggingface_hub
                huggingface_hub.snapshot_download = lambda *a, **kw: f"{root}/hub"
            if killer:
                install(killer)
            try:
                runpy.run_path(f"{ctx.prep}/{script}", run_name="__main__")
            except SystemExit as e:
                if e.code not in (None, 0):
                    raise
            return killer.n if killer else 0

        before = snapshot(root) if check else None
        code, n = fork(main, ctx.log)
        if code == KILLED and killer:
            return False
        if code:
            raise Stop(f"{script} exits {code}: " + " | ".join(open(ctx.log).read().split("\n")[-4:-1]))
        seen[name] = seen.get(name, 0) + int(n)
        bad = backups(before, snapshot(root)) if check else []
        if bad:
            raise Stop(f"{script}: " + "; ".join(bad))
        return True

    step.seen = seen
    return step


def state(ctx, d):
    r = subprocess.run([sys.executable, "-", d], input=ctx.block, text=True, capture_output=True,
                       env=dict(os.environ, FAST_VARIANT="1", DFLASH2="0"))
    if r.returncode:
        raise Stop(f"state() exits {r.returncode}: {r.stderr.strip().splitlines()[-1:]}")
    if "download" in r.stdout.split():
        raise Stop("state() asks to download a directory prepare wrote itself")
    return r.stdout.split()


def prepare_pass(ctx, root, step):
    """One docker/prepare.sh pass. False when the kill fired."""
    d = f"{root}/base"
    scripts = {"lm_head": ("quant_lm_head.py", [d]), "embed": ("quant_embed.py", [d]),
               "mtp": ("quant_mtp.py", [d]), "draft": ("build_draft_vocab.py", [d, "--ids", ctx.ids]),
               "fast": ("fetch_fast_variant.py", [d, d + "-fast"])}
    todo = [s for s in state(ctx, d) if s in scripts]
    dirs = [d] + ([d + "-fast"] if "fast" in todo or os.path.isdir(d + "-fast") else [])
    for s, script, args in [(s, *scripts[s]) for s in todo] + [
            ("harden", "harden_chat_template.py", dirs), ("translate", "translate_chat_template.py", dirs)]:
        if not step(s, script, args):
            return False
    left = [s for s in state(ctx, d) if s in scripts]
    if left:
        raise Stop(f"steps still missing after the pass: {left}")
    return True


def stream_pass(ctx, root, step):
    return step("stream", "quant_heads_stream.py", [f"{root}/single"])


def snapshot(root):
    out = {}
    for base, _, files in os.walk(root):
        for n in files:
            if not n.endswith(".tmp"):
                with open(f"{base}/{n}", "rb") as f:
                    out[os.path.relpath(f"{base}/{n}", root)] = hashlib.sha256(f.read()).hexdigest()
    return out


def backups(before, after):
    """Backups a step created or changed that are not the file as it was before the step."""
    bad = []
    for p, h in sorted(after.items()):
        sfx = next((s for s in BACKUPS if p.endswith(s)), None)
        if sfx and before.get(p) != h:
            if p in before:
                bad.append(f"{p} overwritten")
            elif before.get(p[: -len(sfx)], h) != h:
                bad.append(f"{p} is not {p[: -len(sfx)]} as it was before the step")
    return bad


def check(ctx, name, run_pass, inject):
    """The run without a kill, then one killed and one clean pass per mutation and mode."""
    ref = f"{ctx.work}/{name}-ref"
    shutil.copytree(ctx.pristine, ref)
    step = stepper(ctx, ref, inject, check=True)
    try:
        run_pass(ctx, ref, step)
    except Stop as e:
        print(f"FAIL  {name} without a kill: {e}")
        return 1, 1
    want, total = snapshot(ref), sum(step.seen.values())
    idle = [s for s in inject if not step.seen.get(s)]
    print(f"{name}: {total} file mutations in {', '.join(inject)}")
    fails = cases = 0
    if idle:
        print(f"FAIL  {name}: {', '.join(idle)} wrote nothing, so nothing was tested")
        fails = cases = 1
    for torn in (False, True):
        for at in range(1, total + 1):
            root, label = f"{ctx.work}/{name}-{at}", f"{name} {'torn' if torn else 'after'}@{at}"
            shutil.copytree(ctx.pristine, root)
            cases += 1
            try:
                if run_pass(ctx, root, stepper(ctx, root, inject, at, torn)):
                    raise Stop("the kill never fired")
                run_pass(ctx, root, stepper(ctx, root))
                got = snapshot(root)
                diff = sorted(k for k in got.keys() | want.keys() if got.get(k) != want.get(k))
                if diff:
                    raise Stop(f"the next start completes, but these files differ: {diff}")
                print(f"OK    {label}")
            except Stop as e:
                print(f"FAIL  {label}: {e}")
                fails += 1
            shutil.rmtree(root)
    return fails, cases


def main():
    steps = sys.argv[1:] or list(STEPS)
    if set(steps) - set(STEPS) or not hasattr(os, "fork"):
        sys.exit(f"usage: test_prepare_crash.py [{' '.join(STEPS)}] (Linux only)")
    # Import what the steps import once, before the forks: a cold compressed_tensors
    # import costs each step ~25 s.
    import huggingface_hub, safetensors.torch, tokenizers, torch  # noqa: F401
    import compressed_tensors.compressors.pack_quantized.base  # noqa: F401
    from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: F401

    fails = cases = 0
    with tempfile.TemporaryDirectory(prefix="prepare-crash-") as work:
        # A private copy of prepare/: harden_chat_template.py also rewrites the templates
        # under models/ next to the scripts, which must never reach a real checkout.
        ctx = SimpleNamespace(work=work, prep=f"{work}/repo/prepare", ids=f"{work}/ids.json",
                              log=f"{work}/step.log", pristine=f"{work}/pristine",
                              block=(REPO / "docker/prepare.sh").read_text(encoding="utf-8").split(
                                  'python - "$BASE" <<\'EOF\'\n', 1)[1].split("\nEOF", 1)[0])
        shutil.copytree(REPO / "prepare", ctx.prep)
        if fork(lambda: build(ctx.pristine, ctx.prep, ctx.ids), ctx.log)[0]:
            sys.exit("fixture build failed:\n" + open(ctx.log).read())
        for name, run_pass, own in (("prepare", prepare_pass, STEPS[:-1]), ("stream", stream_pass, STEPS[-1:])):
            inject = [s for s in steps if s in own]
            if inject:
                f, c = check(ctx, name, run_pass, inject)
                fails, cases = fails + f, cases + c
    print(f"{cases} cases, {fails} failures")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
