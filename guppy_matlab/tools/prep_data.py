"""Build guppylm corpus + BPE tokenizer assets for the MATLAB port.

Outputs (all in ../data):
  train_ids.u16   uint16 little-endian, (N, 128) row-major, 0 = pad
  train_len.u16   uint16, (N, 1)  true token length before padding
  eval_ids.u16 / eval_len.u16  same for the held-out split
  bpe_assets.matfrag.json  vocab + merges, consumed by build_bpe_mat.m
  prep_report.txt
"""
import json
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAXLEN = 128
TEMPLATE = (
    "<|im_start|>user\n{inp}<|im_end|>\n"
    "<|im_start|>assistant\n{out}<|im_end|>"
)
SPECIALS = {"<|im_start|>": 1, "<|im_end|>": 2}
SPECIAL_RE = re.compile(r"(<\|im_start\|>|<\|im_end\|>)")
PAT = re.compile(
    r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = bytes_to_unicode()
SPECIAL_BYTE_CHARS = set("".join(B2U[b] for b in range(256)))


def load_bpe(path):
    j = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = j["model"]["vocab"]
    merges = j["model"]["merges"]
    if merges and isinstance(merges[0], str):
        merges = [tuple(m.split(" ")) for m in merges]
    else:
        merges = [tuple(m) for m in merges]
    rank = {p: i for i, p in enumerate(merges)}
    return vocab, merges, rank


def bpe_word(sym, rank, cache):
    """Merge a list of single chars using the learned rank table."""
    key = "".join(sym)
    hit = cache.get(key)
    if hit is not None:
        return hit
    sym = list(sym)
    while len(sym) > 1:
        best_r, best_i = None, None
        for i in range(len(sym) - 1):
            r = rank.get((sym[i], sym[i + 1]))
            if r is not None and (best_r is None or r < best_r):
                best_r, best_i = r, i
        if best_r is None:
            break
        sym = sym[:best_i] + [sym[best_i] + sym[best_i + 1]] + sym[best_i + 2:]
    cache[key] = sym
    return sym


def encode_ordinary(text, vocab, rank, cache):
    ids = []
    for piece in PAT.findall(text):
        sym = bpe_word([B2U[b] for b in piece.encode("utf-8")], rank, cache)
        for s in sym:
            tid = vocab.get(s)
            if tid is None:
                # unseen symbol: fall back to byte-level chars
                for ch in s:
                    ids.append(vocab[ch])
            else:
                ids.append(tid)
    return ids


def encode(text, vocab, rank, cache):
    ids = []
    for p in SPECIAL_RE.split(text):
        if not p:
            continue
        if p in SPECIALS:
            ids.append(SPECIALS[p])
        else:
            ids += encode_ordinary(p, vocab, rank, cache)
    return ids


def decode(ids, inv):
    out = []
    for i in ids:
        if i == 0:
            continue
        if i == 1:
            out.append("<|im_start|>")
            continue
        if i == 2:
            out.append("<|im_end|>")
            continue
        out.append(inv[i])
    s = "".join(out)
    return bytearray(B2U_REV[ch] for ch in s).decode("utf-8", errors="replace")


B2U_REV = {v: k for k, v in B2U.items()}


def pack(seqs, name):
    n = len(seqs)
    buf = bytearray(n * MAXLEN * 2)
    lens = []
    for r, ids in enumerate(seqs):
        ids = ids[:MAXLEN]
        lens.append(len(ids))
        base = r * MAXLEN * 2
        for c, t in enumerate(ids):
            off = base + c * 2
            buf[off] = t & 0xFF
            buf[off + 1] = (t >> 8) & 0xFF
    (DATA / f"{name}_ids.u16").write_bytes(bytes(buf))
    (DATA / f"{name}_len.u16").write_bytes(b"".join(
        bytes([l & 0xFF, (l >> 8) & 0xFF]) for l in lens))
    return n, lens


def main():
    t0 = time.time()
    vocab, merges, rank = load_bpe(DATA / "tokenizer.json")
    inv = {v: k for k, v in vocab.items()}
    print(f"vocab={len(vocab)} merges={len(merges)}")

    tbl = pq.read_table(DATA / "train.parquet").to_pylist()
    print(f"rows={len(tbl)}")

    cache = {}
    texts = [TEMPLATE.format(inp=r["input"].strip(), out=r["output"].strip())
             for r in tbl]

    # round-trip sanity on 20 samples before doing the whole corpus
    for t in texts[:20]:
        ids = encode(t, vocab, rank, cache)
        if decode(ids, inv) != t:
            print("ROUNDTRIP FAIL:", repr(t)[:120])
            sys.exit(1)
    print("roundtrip ok on 20 samples")

    t1 = time.time()
    enc = []
    for i, t in enumerate(texts):
        enc.append(encode(t, vocab, rank, cache))
        if (i + 1) % 10000 == 0:
            print(f"  encoded {i+1} ({time.time()-t1:.1f}s)")
    print(f"encode done {time.time()-t1:.1f}s, cache={len(cache)}")

    lens = [min(len(x), MAXLEN) for x in enc]
    n_eval = 3000
    tr, ev = enc[:-n_eval], enc[-n_eval:]
    n1, l1 = pack(tr, "train")
    n2, l2 = pack(ev, "eval")

    import statistics
    st = {
        "vocab_size": len(vocab),
        "merges": len(merges),
        "train_rows": n1,
        "eval_rows": n2,
        "len_mean": round(statistics.mean(l1), 2),
        "len_p50": statistics.median(l1),
        "len_p95": sorted(l1)[int(0.95 * len(l1))],
        "len_max": max(l1),
        "truncated": sum(1 for x in enc if len(x) > MAXLEN),
        "total_tokens": sum(l1),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (DATA / "prep_report.txt").write_text(json.dumps(st, indent=2), encoding="utf-8")
    print(json.dumps(st, indent=2))

    assets = {
        "vocab": [[k, v] for k, v in vocab.items()],
        "merges": [[a, b] for a, b in merges],
        "byte_chars": [B2U[b] for b in range(256)],
        "maxlen": MAXLEN,
    }
    (DATA / "bpe_assets.json").write_text(
        json.dumps(assets, ensure_ascii=False), encoding="utf-8")
    print("wrote bpe_assets.json")


if __name__ == "__main__":
    main()
