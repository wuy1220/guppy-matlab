"""Build prompt token ids for sampling, and decode generated ids back to text.

Usage:
  python tools/prompts.py build     -> data/prompts.txt, data/prompt_texts.txt
  python tools/prompts.py decode    -> reads data/gen_<tag>.txt, prints text
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PROMPTS = [
    "hi guppy",
    "are you hungry",
    "do you like bubbles",
    "what is the meaning of life",
    "tell me a joke",
    "do you love me",
    "it is raining outside",
    "the cat is looking at you",
    "goodnight guppy",
    "are you scared of the filter",
]

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_data import B2U, load_bpe, encode, decode, B2U_REV  # noqa: E402


def fmt(msg):
    return f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"


def build():
    vocab, _, rank = load_bpe(DATA / "tokenizer.json")
    cache = {}
    lines, texts = [], []
    for m in PROMPTS:
        t = fmt(m)
        ids = encode(t, vocab, rank, cache)
        lines.append(",".join(str(i) for i in ids))
        texts.append(t)
    (DATA / "prompts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DATA / "prompt_texts.txt").write_text("\n".join(texts), encoding="utf-8")
    print(f"wrote {len(lines)} prompts, lens {[len(l.split(',')) for l in lines]}")


def decode_file(path):
    vocab, _, _ = load_bpe(DATA / "tokenizer.json")
    inv = {v: k for k, v in vocab.items()}
    prompts = [fmt(m) for m in PROMPTS]
    lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
    for i, ln in enumerate(lines):
        ids = [int(x) for x in ln.split(",") if x != ""]
        txt = decode(ids, inv)
        q = prompts[i].split("user\n")[1].split("<|im_end|>")[0] if i < len(prompts) else "?"
        ans = txt.split("assistant\n", 1)[-1] if "assistant\n" in txt else txt
        ans = ans.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        print(f"You> {q}")
        print(f"Guppy> {ans}\n")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        decode_file(sys.argv[2])
