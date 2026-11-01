#!/usr/bin/env python3
"""Find calls our series adds that no longer bind to the signature the target pin gives their callee.

    python scripts/port-changed-signatures.py --fork ../vllm --old v0.30.0 --new v0.31.0rc3 --branch qwen38/0.31-rc3

Born 2026-10-02 from syv-ai #261. vLLM #52188 (in 0.29.0) added cp_rank, cp_size and cp_interleave to
prepare_dflash_inputs and updated its own caller. The series' second caller, in dflash2-ngram-chains, was not updated,
the patch still applied cleanly, and VLLM_DFLASH2_CHAIN=1 failed on the first request with a TypeError (booted on 0.30;
0.29 read from the code). Neither a replay nor port-removed-names can see that: the name still exists, only its
parameters changed.

Call sites: every call that touches a line the series adds (the net diff from whichever of --new / --old the branch
contains, so this runs before a rebase as a preview and after it as the check). Each callee is RESOLVED before it is
bound, the way Python would find it: a bare name through the file's own definitions and its `from vllm... import`s; a
`module.func(...)` through an imported vllm module; `self.method(...)` / `cls.method(...)` through the enclosing class
and its bases; `Class.method(...)` and `Class(...)` (its __init__) through the named class; a Triton launch
`kernel[grid](...)` as a call to the kernel, less Triton's own launch options (num_warps and the like), with the
parameters @triton.autotune / @triton.heuristics supply left optional. Any other receiver (`tensor.view`, `logger.info`,
`torch.cat`, `self.model.forward`) has a type this cannot know and is skipped, as are calls that spread *args or
**kwargs; on the rc3 port that is ~2,300 skipped against ~185 bound, most of them torch, triton.language, builtins and
locals. Definitions are the --new tag's, except those the series itself adds or re-signs (signature lines in the
series' added lines, or a file the series adds), which are read from the branch, so a signature the series changes is
the one a call is held to. Binding follows Python's rules: positional count, keyword names, required parameters,
*args / **kwargs on the def. A constructor reached through a @support_torch_compile class is bound the way that
decorator's wrapper binds it: vllm_config= and prefix= are the wrapper's own (dropped when the original __init__ lacks
them, re-injected as keywords when it has them, so passing either positionally is "multiple values"). Its positional
isinstance check is not modelled; it is off by one on vLLM main at writing (it zips args against a parameter list
that includes self), so a positional construction of a decorated class can raise with correct types.

  BREAKS   the resolved definition does not accept the call: a TypeError the moment the path runs

Each hit names the series topic that owns the call's line (by blame) and where the callee's signature comes from:
"changed <old>..<new>" (upstream moved it under us during this port), "same in <old> and <new>" (it was already
broken before this port), or "the series' own definition" (we re-signed it and missed a caller). --accept NAME=REASON
lists a reviewed call as accepted; --verbose lists every call that binds. Exit 1 on any unaccepted BREAKS.

The falsifier, 2026-10-02 (ten runs, each as predicted). Red, the chains call alone: the 0.30 series before #261,
the 0.31rc3 port before its chains fix, and a preview of the 0.30 series against v0.31.0rc3. Green: both after the
fix. Historical: at the 0.29 port, the preview (the 0.28 series against v0.29.0) reports all three series callers
#52188 changed, "changed v0.28.0..v0.29.0" (the chains call, the lookup-drafting call and the prewarm topic's kernel
launch), and the post-rebase run reports the one the rebase left stale, chains. Mutants of the fixed rc3 series built
with plumbing: a keyword renamed through an import, a positional dropped from a constructor, an extra positional on a
self. method, a series classmethod re-signed (two callers), a series kernel re-signed and an unknown keyword on a
launch went red on exactly those seven calls and nothing else. The decorated constructor, both ways (a positional
prefix must break, a vllm_config= to a class without one must not), was wrong in the first cut and is right now,
matching vLLM's own wrapper run on the rc3 tree (vllm-project/vllm#60202 has the type-check bug it turned up).
Reviewed 2026-10-02 by binding every call this check binds with inspect.signature on an installed rc3 tree (185 of
185 agree) and by hunting the skipped calls for a miss (none found; the names whose parameters changed this port were
bound by hand).
"""
import argparse
import ast
import re
import subprocess
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
MAX_DEPTH = 6
# @support_torch_compile replaces a class's __init__ with (*args, vllm_config=None, prefix="", **kwargs) and passes
# vllm_config / prefix on to the original as keywords only when the original takes them (vllm/compilation/decorators.py).
STC_KEYWORDS = ("vllm_config", "prefix")
TRITON_LAUNCH_OPTIONS = {"num_warps", "num_stages", "num_ctas", "maxnreg", "enable_fp_fusion", "waves_per_eu",
                         "matrix_instr_nonkdim", "kpack", "launch_cooperative_grid", "launch_pdl", "warmup", "debug",
                         "num_buffers_warp_spec", "num_consumer_groups", "reg_dec_producer", "reg_inc_consumer"}


def git(fork, *a):
    return subprocess.run(["git", "-C", fork, *a], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def contains(fork, ancestor, rev):
    return subprocess.run(["git", "-C", fork, "merge-base", "--is-ancestor", ancestor, rev]).returncode == 0


class Blobs:
    """Lazy reads of rev:path through one `git cat-file --batch` pipe, cached."""

    def __init__(self, fork):
        self.p = subprocess.Popen(["git", "-C", fork, "cat-file", "--batch"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)
        self.cache = {}

    def get(self, rev, path):
        k = (rev, path)
        if k not in self.cache:
            self.p.stdin.write(f"{rev}:{path}\n".encode())
            self.p.stdin.flush()
            header = self.p.stdout.readline().decode().split()
            if len(header) < 3 or header[1] != "blob":
                self.cache[k] = None
            else:
                self.cache[k] = self.p.stdout.read(int(header[2])).decode("utf-8", errors="replace")
                self.p.stdout.read(1)
        return self.cache[k]


class Sig:
    def __init__(self, node, is_method, where):
        a = node.args
        self.is_method = is_method
        self.where = where
        decos = set()
        for d in node.decorator_list:
            d = d.func if isinstance(d, ast.Call) else d
            decos.add(d.id if isinstance(d, ast.Name) else getattr(d, "attr", ""))
        self.static = "staticmethod" in decos
        self.classmethod = "classmethod" in decos
        # @triton.autotune / @triton.heuristics supply some parameters themselves, so a launch may omit them.
        self.tuned = bool(decos & {"autotune", "heuristics"})
        self.ours = False  # set when the definition is read from the branch as the series' own
        self.posonly = [x.arg for x in a.posonlyargs]
        self.params = [x.arg for x in a.posonlyargs + a.args]
        self.ndefaults = len(a.defaults)
        self.kwonly = [x.arg for x in a.kwonlyargs]
        self.kwonly_required = [x.arg for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is None]
        self.vararg = a.vararg.arg if a.vararg else None
        self.kwarg = a.kwarg.arg if a.kwarg else None
        self.lines = (node.lineno, node.body[0].lineno - 1 if node.body else node.lineno)

    def key(self):
        return (tuple(self.params), self.ndefaults, tuple(self.kwonly), tuple(self.kwonly_required),
                bool(self.vararg), bool(self.kwarg))

    def text(self):
        star = [f"*{self.vararg}"] if self.vararg else ["*"] if self.kwonly else []
        return f"({', '.join(self.params + star + self.kwonly + ([f'**{self.kwarg}'] if self.kwarg else []))})"

    def binds(self, npos, kwnames, bound, launch=False, stc=False):
        params = list(self.params)
        nreq = len(params) - self.ndefaults
        if bound and self.is_method and not self.static and params:
            params, nreq = params[1:], nreq - 1
        if stc:  # the wrapper takes these itself, drops them, or re-injects them as keywords
            kwnames = [k for k in kwnames if k not in STC_KEYWORDS] + [
                k for k in STC_KEYWORDS if k in params or k in self.kwonly]
        if launch:  # kernel[grid](...): the launch options are Triton's, not the kernel's parameters
            kwnames = [k for k in kwnames if k in params or k in self.kwonly or k not in TRITON_LAUNCH_OPTIONS]
            if self.tuned:
                nreq = 0
        if npos > len(params) and not self.vararg:
            return False
        filled = set(params[:npos])
        for k in kwnames:
            if k in filled or k in self.posonly:
                return False
            if k in params or k in self.kwonly:
                filled.add(k)
            elif not self.kwarg:
                return False
        return all(r in filled for r in params[:max(nreq, 0)]) and all(k in filled for k in self.kwonly_required)


class Module:
    """Top-level functions, classes (with bases and methods) and import bindings of one file."""

    def __init__(self, path, src):
        self.path = path
        self.funcs, self.classes, self.names, self.modules = {}, {}, {}, {}
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = ast.Module(body=[], type_ignores=[])
        pkg = path.rsplit("/", 1)[0].replace("/", ".")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs[node.name] = Sig(node, False, f"{path}:{node.lineno}")
            elif isinstance(node, ast.ClassDef):
                methods = {s.name: Sig(s, True, f"{path}:{s.lineno} ({node.name})") for s in node.body
                           if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))}
                bases = [b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else None
                         for b in node.bases]
                decos = {(d.func if isinstance(d, ast.Call) else d) for d in node.decorator_list}
                self.classes[node.name] = {"bases": [b for b in bases if b], "methods": methods,
                                           "decos": {d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
                                                     for d in decos}}
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._imports(node, pkg)
            elif isinstance(node, ast.If):  # TYPE_CHECKING blocks and similar
                for sub in node.body:
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        self._imports(sub, pkg)

    def _imports(self, node, pkg):
        if isinstance(node, ast.Import):
            for al in node.names:
                if al.asname:
                    self.modules[al.asname] = al.name
        else:
            mod = node.module or ""
            if node.level:
                parts = pkg.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                mod = f"{base}.{mod}" if mod else base
            for al in node.names:
                self.names[al.asname or al.name] = (mod, al.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--accept", action="append", default=[], metavar="NAME=REASON")
    ap.add_argument("--verbose", action="store_true", help="also list every call that binds, and where its callee was found")
    a = ap.parse_args()
    a.accept = dict(x.split("=", 1) for x in a.accept)

    if contains(a.fork, a.new, a.branch):
        base, mode = a.new, "after the rebase"
    elif contains(a.fork, a.old, a.branch):
        base, mode = a.old, "before the rebase (a preview against the new pin)"
    else:
        sys.exit(f"{a.branch} contains neither {a.new} nor {a.old}")

    added = defaultdict(set)  # the series' added lines per file, at the branch tip (net diff)
    f = None
    for ln in git(a.fork, "diff", "-U0", base, a.branch, "--", "*.py").splitlines():
        if ln.startswith("+++ "):
            f = ln[6:] if ln != "+++ /dev/null" else None
        elif ln.startswith("@@") and f:
            m = re.search(r"\+(\d+)(?:,(\d+))?", ln)
            start, count = int(m.group(1)), int(m.group(2) or 1)
            added[f].update(range(start, start + count))
    touched = sorted(p for p in added if p.startswith("vllm/") and p.endswith(".py"))

    blobs = Blobs(a.fork)
    mods = {}

    def module(rev, path):
        k = (rev, path)
        if k not in mods:
            src = blobs.get(rev, path)
            mods[k] = Module(path, src) if src is not None else None
        return mods[k]

    def target(path):
        """The module as the target pin has it, with the series' own definitions from the branch laid over it."""
        k = ("target", path)
        if k in mods:
            return mods[k]
        m = module(a.new, path)
        if path in added:
            b = module(a.branch, path)
            if b is not None:
                if m is None:  # a file the series adds: every definition in it is ours
                    m = b
                    for sig in list(b.funcs.values()) + [s for c in b.classes.values() for s in c["methods"].values()]:
                        sig.ours = True
                else:
                    ours = added[path]
                    o = module(a.old, path)

                    def is_ours(sig):
                        return any(n in ours for n in range(sig.lines[0], sig.lines[1] + 1))

                    def series_only(old_has):
                        # Absent from --new and present on the branch: the series' own after the rebase. Before it,
                        # the branch is --old's tree, so a definition --old has is one upstream deleted, and laying
                        # it back would hide that deletion (port-removed-names reports deleted names).
                        return base == a.new or not old_has
                    for name, sig in b.funcs.items():
                        if is_ours(sig) or (name not in m.funcs and series_only(o is not None and name in o.funcs)):
                            sig.ours = True
                            m.funcs[name] = sig
                    for cname, c in b.classes.items():
                        oc = o.classes.get(cname) if o is not None else None
                        if cname not in m.classes and not series_only(oc is not None):
                            continue
                        tc = m.classes.setdefault(cname, {"bases": c["bases"], "methods": {}, "decos": c["decos"]})
                        for mname, sig in c["methods"].items():
                            if is_ours(sig) or (mname not in tc["methods"]
                                                and series_only(oc is not None and mname in oc["methods"])):
                                sig.ours = True
                                tc["methods"][mname] = sig
                    m.names.update({k2: v for k2, v in b.names.items() if k2 not in m.names})
                    m.modules.update({k2: v for k2, v in b.modules.items() if k2 not in m.modules})
        mods[k] = m
        return m

    def mod_file(modname):
        if not modname.startswith("vllm"):
            return None
        stem = modname.replace(".", "/")
        for cand in (f"{stem}.py", f"{stem}/__init__.py"):
            if blobs.get(a.new, cand) is not None or blobs.get(a.branch, cand) is not None:
                return cand
        return None

    def resolve_name(path, name, depth=0):
        """('func', Sig, path) / ('class', path, cname) / None for a name visible at module scope of path."""
        m = target(path)
        if m is None or depth > MAX_DEPTH:
            return None
        if name in m.funcs:
            return ("func", m.funcs[name], path)
        if name in m.classes:
            return ("class", path, name)
        if name in m.names:
            mod, sym = m.names[name]
            fp = mod_file(mod)
            if fp:
                r = resolve_name(fp, sym, depth + 1)
                if r:
                    return r
            sub = mod_file(f"{mod}.{sym}")  # `from vllm.x import module`
            if sub:
                return ("module", sub)
        return None

    def resolve_method(path, cname, mname, depth=0):
        m = target(path)
        if m is None or cname not in m.classes or depth > MAX_DEPTH:
            return None
        c = m.classes[cname]
        if mname in c["methods"]:
            return c["methods"][mname]
        for bname in c["bases"]:
            r = resolve_name(path, bname)
            if r and r[0] == "class":
                hit = resolve_method(r[1], r[2], mname, depth + 1)
                if hit:
                    return hit
        return None

    def init_of(path, cname, depth=0):
        """(the __init__ a `Class(...)` call reaches, whether a @support_torch_compile class on the way wraps it)."""
        m = target(path)
        if m is None or cname not in m.classes or depth > MAX_DEPTH:
            return None, False
        c = m.classes[cname]
        stc = "support_torch_compile" in c.get("decos", ())
        if "__init__" in c["methods"]:
            return c["methods"]["__init__"], stc
        for bname in c["bases"]:
            r = resolve_name(path, bname)
            if r and r[0] == "class":
                sig, wrapped = init_of(r[1], r[2], depth + 1)
                if sig:
                    return sig, wrapped or stc
        return None, False

    def changed_since_old(sig, path, cname, fname):
        m = module(a.old, path)
        if m is None:
            return True
        old = m.classes.get(cname, {}).get("methods", {}).get(fname) if cname else m.funcs.get(fname)
        return old is None or old.key() != sig.key()

    def topic(path, n):
        out = git(a.fork, "blame", "--porcelain", "-L", f"{n},{n}", a.branch, "--", path)
        return next((l[8:] for l in out.splitlines() if l.startswith("summary ")), "?")

    breaks = checked = skipped = 0
    for path in touched:
        src = blobs.get(a.branch, path)
        try:
            tree = ast.parse(src or "")
        except SyntaxError as e:
            print(f"SYNTAX   {path}: {e}")
            breaks += 1
            continue
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            span = range(node.lineno, (node.end_lineno or node.lineno) + 1)
            if not any(n in added[path] for n in span):
                continue
            if any(isinstance(x, ast.Starred) for x in node.args) or any(k.arg is None for k in node.keywords):
                skipped += 1
                continue
            fn, sig, bound, owner, stc = node.func, None, False, (None, None), False
            launch = isinstance(fn, ast.Subscript)
            if launch:
                fn = fn.value
            if isinstance(fn, ast.Name):
                r = resolve_name(path, fn.id)
                if r and r[0] == "func":
                    sig, owner = r[1], (r[2], None)
                elif r and r[0] == "class":
                    (sig, stc), bound, owner = init_of(r[1], r[2]), True, (r[1], r[2])
                name = fn.id
            elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
                recv, name = fn.value.id, fn.attr
                fdef = parents.get(node)
                while fdef is not None and not isinstance(fdef, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fdef = parents.get(fdef)
                cls = parents.get(fdef) if fdef is not None else None
                first = fdef.args.args[0].arg if fdef is not None and fdef.args.args else None
                if isinstance(cls, ast.ClassDef) and recv == first and recv in ("self", "cls"):
                    sig, bound, owner = resolve_method(path, cls.name, name), True, (path, cls.name)
                else:
                    r = resolve_name(path, recv)
                    m = target(path)
                    modname = m.modules.get(recv) if m else None
                    if r and r[0] == "class":
                        sig, owner = resolve_method(r[1], r[2], name), (r[1], r[2])
                        # Class.method(...): a classmethod binds cls; a plain method takes self explicitly.
                        bound = sig is not None and sig.is_method and sig.classmethod
                    elif r and r[0] == "module":
                        rr = resolve_name(r[1], name)
                        if rr and rr[0] == "func":
                            sig, owner = rr[1], (rr[2], None)
                    elif modname and mod_file(modname):
                        rr = resolve_name(mod_file(modname), name)
                        if rr and rr[0] == "func":
                            sig, owner = rr[1], (rr[2], None)
            else:
                name = None
            if sig is None:
                skipped += 1
                continue
            checked += 1
            npos, kws = len(node.args), [k.arg for k in node.keywords]
            if sig.binds(npos, kws, bound, launch, stc):
                if a.verbose:
                    call = f"{name}{'[grid]' if launch else ''}({npos} positional{', ' + ', '.join(kws) if kws else ''})"
                    print(f"ok       {path}:{node.lineno}: {call} -> {sig.where} {sig.text()}")
                continue
            if name in a.accept:
                print(f"accepted {path}:{node.lineno}: {name} ({a.accept[name]})")
                continue
            breaks += 1
            def_path = sig.where.split(":", 1)[0]
            cname = sig.where.split("(", 1)[1].rstrip(")") if "(" in sig.where else None
            fname = "__init__" if (isinstance(fn, ast.Name) and owner[1]) else name
            if sig.ours:
                when = "the series' own definition"
            elif changed_since_old(sig, def_path, cname, fname):
                when = f"changed {a.old}..{a.new}"
            else:
                when = f"same in {a.old} and {a.new}"
            call = f"{name}{'[grid]' if launch else ''}({npos} positional{', ' + ', '.join(kws) if kws else ''})"
            ours_line = min(n for n in span if n in added[path])  # blame the series' line, not the call's first
            print(f"BREAKS   {topic(path, ours_line)[:50]:50}  {path}:{node.lineno}: {call}  [{when}]")
            print(f"         takes {sig.where} {sig.text()}"
                  + ("  through @support_torch_compile (vllm_config / prefix go by keyword)" if stc else ""))
    print(f"port-changed-signatures {a.old} -> {a.new}, {mode}: {checked} resolved calls on the series' added lines in "
          f"{len(touched)} files bound against {a.new}; {skipped} skipped (receiver type unknown, or *args/**kwargs "
          f"spread); {breaks} BREAKS")
    sys.exit(1 if breaks else 0)


if __name__ == "__main__":
    main()
