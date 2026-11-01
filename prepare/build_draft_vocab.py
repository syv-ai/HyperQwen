"""Build a vocab-truncated draft head for MTP speculative decoding.

The MTP drafter has to run the 248k-row lm_head once per draft token, and at
4-6 drafts per step that head read (1.3 GB int8) dominates the draft cost.
Speculative decoding stays exact no matter what the drafter proposes, so the
drafter can use a head restricted to the N most frequent tokens: tokens
outside the shortlist are simply never drafted (that position gets rejected
and the target's own sample is used, as always).

This script counts token frequencies over a text corpus (Danish + English +
code + the model's own outputs), picks the top N ids (plus special tokens),
slices those rows out of the already-int8-quantized lm_head, and stores them
as `mtp.draft_lm_head.*` in model_extra_tensors.safetensors, plus the id map in
`mtp_draft_vocab_ids.pt`. Needs the matching vLLM patch
(patches/qwen3_5-mtp-draft-vocab.patch) to be used.

Usage (from the repo root):
  python prepare/build_draft_vocab.py /path/to/model --ids prepare/draft_vocab_ids.json  # shipped id list
  python prepare/build_draft_vocab.py /path/to/model --n 40960 --corpus f1 f2 ...        # or count your own
  python prepare/build_draft_vocab.py /path/to/model --stats                           # coverage per script
  python prepare/build_draft_vocab.py /path/to/model --ids prepare/draft_vocab_ids.json \
      --add-scripts cjk --script-budget 16384 --corpus zh.txt --out cjk.json            # variant id list
Corpus files: .txt/.jsonl (uses "prompt"/"response"/"messages"/"text" fields)/.parquet(text)/.py
The shipped draft_vocab_ids.json was counted over Danish web text (fineweb-2),
English Wikipedia, Python source and the model's own chat outputs (8.8M tokens);
held-out coverage 95%.

--stats and --add-scripts exist because that list is language-specific, and not
in a small way: it holds 3 of this vocabulary's 55,328 Han ids and 1 of its
18,580 Cyrillic ids (#196, gotcha 61). --stats prints the coverage per script so
that is measured rather than described. --add-scripts writes a *variant* id
list -- the base ids unchanged and in their order, whole-UTF-8 rows of the named
scripts appended, ranked by the corpus counts and capped at --script-budget rows
-- and then stops without touching the model. Build the variant head from it in
a second run:

  python prepare/build_draft_vocab.py /path/to/model --ids cjk.json

That keeps the cap explicit and the shipped default untouched: on this
checkpoint a head row is 2,560 B, so an uncapped CJK union is +168.8 MB of VRAM
against +41 MB for the default 16,384-row budget.

Every output is written through prepare/atomic_publish.py (a temp file and a rename),
and the index goes last, after the extras shard and mtp_draft_vocab_ids.pt: the index
is what activates the draft head (docker/prepare.sh's state() reads it), so it must not
point at a head whose id list is not on disk yet. A killed run is completed by the
next one (#195).
"""
import glob, json, os, sys, collections
import torch
from safetensors import safe_open

from atomic_publish import backup_once, publish, save_tensors, write_json
from quant_schema import index_packed

def argval(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

d = sys.argv[1].rstrip("/") + "/"
N = int(argval("--n", 40960))
corpus = sys.argv[sys.argv.index("--corpus") + 1:] if "--corpus" in sys.argv else []
# --corpus takes everything after it, so a later flag would land in the file
# list. Stop at the next flag: --corpus may be written before --add-scripts.
if corpus:
    corpus = [a for a in corpus[:next((k for k, a in enumerate(corpus)
                                        if a.startswith("--")), len(corpus))]]
ids_file = argval("--ids")
stats = "--stats" in sys.argv
add_scripts = argval("--add-scripts", "")
script_budget = int(argval("--script-budget", 16384))
out_file = argval("--out")

# Unicode ranges per script. A draft-vocab row must be a token the model can
# emit on its own, so only whole-UTF-8 rows are eligible: a BPE piece holding
# half a multi-byte character is never drafted alone and would waste a head row.
SCRIPTS = {
    "han":       [(0x3400,0x4DBF),(0x4E00,0x9FFF),(0xF900,0xFAFF),(0x20000,0x2FA1F)],
    "kana":      [(0x3040,0x30FF),(0x31F0,0x31FF),(0xFF66,0xFF9F)],
    "hangul":    [(0x1100,0x11FF),(0x3130,0x318F),(0xAC00,0xD7AF)],
    "cjk_punct": [(0x3000,0x303F),(0xFF00,0xFF65),(0xFFA0,0xFFEF)],
    "cyrillic":  [(0x0400,0x04FF),(0x0500,0x052F),(0x1C80,0x1C8F),(0x2DE0,0x2DFF),(0xA640,0xA69F)],
    "latin_ext": [(0x00C0,0x024F),(0x1E00,0x1EFF),(0x2C60,0x2C7F),(0xA720,0xA7FF)],
}
GROUPS = {"cjk": ["han", "kana", "hangul", "cjk_punct"]}

def scripts_of(ch):
    c = ord(ch)
    return {n for n, rs in SCRIPTS.items() if any(a <= c <= b for a, b in rs)}

def bytes_to_unicode():
    # Qwen's tokenizer is byte-level BPE: a vocab key is the token's text with
    # every byte mapped through GPT-2's alphabet, so the key must be inverted
    # before its bytes can be tested for standing alone.
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))

def classify_vocab(model_dir):
    """whole[i] -> the row's bytes stand alone; kindset[i] -> its scripts."""
    uni2byte = {v: k for k, v in bytes_to_unicode().items()}
    vocab = json.load(open(model_dir + "/tokenizer.json"))["model"]["vocab"]
    n = max(vocab.values()) + 1
    whole = bytearray(n)
    kindset = [frozenset()] * n
    nfrag = 0
    for piece, i in vocab.items():
        try:
            raw = bytes(uni2byte[c] for c in piece)
        except KeyError:
            raw = piece.encode("utf-8", errors="ignore")
        if not raw:
            continue
        try:
            txt = raw.decode("utf-8")
        except UnicodeDecodeError:
            nfrag += 1
            continue
        if not txt:
            continue
        whole[i] = 1
        s = set()
        for ch in txt:
            s |= scripts_of(ch)
        kindset[i] = frozenset(s)
    return whole, kindset, nfrag

def expand_scripts(names):
    want = set()
    for x in filter(None, (s.strip() for s in names.split(","))):
        grp = GROUPS.get(x, [x])
        for nm in grp:
            if nm not in SCRIPTS:
                sys.exit(f"--add-scripts {x}: expected a comma list of {', '.join([*SCRIPTS, *GROUPS])}")
            want.add(nm)
    return want

def report(whole, kindset, nrows, ids, nfrag, row_bytes=None):
    print(f"\nvocab rows {nrows}  whole-UTF-8 {sum(whole)} ({100*sum(whole)/nrows:.1f}%)  fragments {nfrag}")
    if ids is not None:
        print(f"id list: {len(ids)} ids\n")
        print("%-11s %8s %10s %12s %11s %10s" % ("script", "vocab", "whole", "in list", "% of script", "% of list"))
        print("-" * 66)
        rows = []
        for name in SCRIPTS:
            tot = sum(1 for i in range(nrows) if name in kindset[i])
            w = sum(1 for i in range(nrows) if name in kindset[i] and whole[i])
            ins = sum(1 for i in ids if i < nrows and name in kindset[i])
            rows.append((name, tot, w, ins))
            print("%-11s %8d %10d %12d %10.1f%% %9.1f%%" % (
                name, tot, w, ins, 100 * ins / max(1, tot), 100 * ins / max(1, len(ids))))
        for gname, members in GROUPS.items():
            grp = [r for r in rows if r[0] in members]
            t, w, i = (sum(x[j] for x in grp) for j in (1, 2, 3))
            print("%-11s %8d %10d %12d %10.1f%% %9.1f%%" % (
                gname.upper() + " total", t, w, i, 100 * i / max(1, t), 100 * i / max(1, len(ids))))
    if row_bytes:
        print(f"\nhead row {row_bytes} B  ->  +16,384 rows = +{16384*row_bytes/1e6:.1f} MB VRAM")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(d)
whole = kindset = None
if stats or add_scripts:
    whole, kindset, nfrag = classify_vocab(d)

def texts_from(path, limit_bytes=20_000_000):
    n = 0
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for t in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            yield t; n += len(t)
            if n > limit_bytes: return
    elif path.endswith(".jsonl"):
        for line in open(path):
            try: r = json.loads(line)
            except Exception: continue
            parts = []
            for k in ("prompt", "response", "text"):
                if isinstance(r.get(k), str): parts.append(r[k])
            if isinstance(r.get("messages"), list):
                parts += [m.get("content", "") for m in r["messages"] if isinstance(m.get("content"), str)]
            t = "\n".join(parts); yield t; n += len(t)
            if n > limit_bytes: return
    else:
        # A plain file is one document, so yield it in ~4 KB runs of lines: the
        # held-out split below takes every 10th text, and a file yielded whole is
        # text 0, which put all of it in the held-out set and left the counts
        # empty (#196).
        buf, size = [], 0
        for line in open(path, errors="ignore"):
            buf.append(line); size += len(line)
            if size >= 4096:
                yield "".join(buf); n += size; buf, size = [], 0
                if n > limit_bytes: return
        if buf:
            yield "".join(buf)

counts = collections.Counter()
held = collections.Counter()
total = 0
base_ids = None
if ids_file:
    base_ids = sorted(set(json.load(open(ids_file))))
    print(f"using {len(base_ids)} ids from {ids_file}")
    # A variant union needs the corpus to rank the added rows by, so the base
    # list no longer cancels it; a plain --ids run still ignores the corpus.
    if not add_scripts:
        corpus = []
ids = base_ids
for i, path in enumerate(corpus):
    for j, t in enumerate(texts_from(path)):
        got = tok(t, add_special_tokens=False).input_ids
        (held if j % 10 == 0 else counts).update(got)
        total += len(got)
print(f"corpus tokens: {total}")

if stats:
    row_bytes = None
    try:
        from safetensors import safe_open as _sf
        _idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
        if "mtp.draft_lm_head.weight_packed" in _idx:
            with _sf(d + _idx["mtp.draft_lm_head.weight_packed"], framework="pt") as _f:
                row_bytes = _f.get_tensor("mtp.draft_lm_head.weight_packed").shape[1] * 4
    except Exception:
        pass
    report(whole, kindset, len(kindset), ids, nfrag, row_bytes)
    if not add_scripts:
        sys.exit(0)

if add_scripts:
    if base_ids is None:
        sys.exit("--add-scripts needs --ids <base list>: a variant is the base list plus rows")
    if not counts and not held:
        sys.exit("--add-scripts needs --corpus to rank the added rows by; "
                 "no corpus tokens were counted and the model dir is unchanged")
    want = expand_scripts(add_scripts)
    have = set(base_ids)
    # Rank by corpus frequency, so the cap buys the rows this traffic actually
    # emits rather than an arbitrary slice of a Unicode range. Both splits
    # count: the held-out 10th exists to report unbiased *coverage*, and a
    # corpus small enough to fit one text (or one 4 KB run) lands entirely in
    # it, so ranking on counts alone would see no corpus at all. Ties by id, so
    # a rebuild from the same corpus is byte-identical.
    seen = collections.Counter(counts)
    seen.update(held)
    cands = [i for i in range(len(kindset))
             if whole[i] and kindset[i] & want and i not in have]
    cands.sort(key=lambda i: (-seen.get(i, 0), i))
    added = cands if script_budget == 0 else cands[:script_budget]
    added.sort()
    variants = base_ids + added
    report(whole, kindset, len(kindset), variants, nfrag)
    print(f"\nunion of {','.join(sorted(want))}: {len(added)} rows added "
          f"({'uncapped, all eligible' if script_budget == 0 else f'budget {script_budget}'}, "
          f"{len(cands)} eligible)")
    if not out_file:
        sys.exit("--add-scripts needs --out <file.json>; the model dir is unchanged")
    with open(out_file + ".tmp", "w") as f:
        json.dump(variants, f)
    publish(out_file + ".tmp", out_file)
    print(f"variant id list written to {out_file} ({len(variants)} ids)")
    print("build the variant head from it in a second run:\n"
          f"  {sys.argv[0]} {d} --ids {out_file}")
    sys.exit(0)

special = set(tok.all_special_ids)
if ids_file:
    special = set()
for name in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"):
    tid = tok.convert_tokens_to_ids(name)
    if isinstance(tid, int) and tid >= 0: special.add(tid)
if not ids_file and not counts:
    sys.exit("no corpus tokens were counted: pass --corpus files with text in them, or --ids; "
             "the model dir is unchanged")
if not ids_file:
    top = [t for t, _ in counts.most_common() if t not in special][: N - len(special)]
    ids = sorted(set(top) | special)
    cover = sum(c for t, c in held.items() if t in set(ids)) / max(1, sum(held.values()))
    print(f"draft vocab: {len(ids)} ids, held-out token coverage {cover*100:.2f}%")
    for n_try in (16384, 32768, 49152, 65536):
        s = set(t for t, _ in counts.most_common(n_try)) | special
        c = sum(c for t, c in held.items() if t in s) / max(1, sum(held.values()))
        print(f"  coverage at N={n_try}: {c*100:.2f}%")
    with open(d + "draft_vocab_ids.json.tmp", "w") as f:
        json.dump(ids, f)
    publish(d + "draft_vocab_ids.json.tmp", d + "draft_vocab_ids.json")
    print(f"id list written to {d}draft_vocab_ids.json (copy it next to this script to reuse)")

# slice lm_head rows
idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
head_shard = wm["lm_head.weight_packed"]
with safe_open(d + head_shard, framework="pt") as f:
    wp = f.get_tensor("lm_head.weight_packed")   # [vocab, K/8] int32
    ws = f.get_tensor("lm_head.weight_scale")    # [vocab, K/group]
    shape = f.get_tensor("lm_head.weight_shape")
ids_t = torch.tensor(ids, dtype=torch.int64)
sub_p = wp.index_select(0, ids_t).contiguous()
sub_s = ws.index_select(0, ids_t).contiguous()
sub_shape = torch.tensor([len(ids), int(shape[1])], dtype=torch.int64)
print(f"draft head: packed {tuple(sub_p.shape)} {sub_p.dtype}, scales {tuple(sub_s.shape)} {sub_s.dtype}, "
      f"{(sub_p.numel()*4 + sub_s.numel()*2)/1e6:.0f} MB")

extra = "model_extra_tensors.safetensors"
# The three quant_*.py scripts leave the base model's extras in this file; a
# single-shard export processed by prepare/quant_heads_stream.py has no extras
# file at all (#37) -- the draft head becomes its first content.
tensors = {}
meta = None
if os.path.exists(d + extra):
    with safe_open(d + extra, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    backup_once(d + extra, ".bak-draft")
tensors["mtp.draft_lm_head.weight_packed"] = sub_p
tensors["mtp.draft_lm_head.weight_scale"] = sub_s
tensors["mtp.draft_lm_head.weight_shape"] = sub_shape
save_tensors(tensors, d + extra, meta or {"format": "pt"})
torch.save(ids_t, d + "mtp_draft_vocab_ids.pt.tmp")
publish(d + "mtp_draft_vocab_ids.pt.tmp", d + "mtp_draft_vocab_ids.pt")
# The index is the commit point, so it goes last.
index_packed(wm, "mtp.draft_lm_head", extra)
write_json(d + "model.safetensors.index.json", idx)
print("done")
