"""Fetch and clean a Malayalam corpus, then encode it to a flat uint16 array.

Source: the `wikimedia/wikipedia` dataset on Hugging Face, config `20231101.ml`
(Malayalam Wikipedia, plain text, ~85k articles). Lines that are not mostly
Malayalam script are dropped so the model spends its capacity on Malayalam
rather than on English names, dates and markup residue.

    python -m mltx.data                 # download + encode
    python -m mltx.data --txt my.txt    # encode your own text file instead
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from .tokenizer import CharTokenizer, normalize

MALAYALAM = re.compile(r"[ഀ-ൿ]")
WS = re.compile(r"[ \t ]+")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def clean_article(text: str, min_ml_ratio: float = 0.6) -> str:
    out = []
    for line in normalize(text).split("\n"):
        line = WS.sub(" ", line).strip()
        if len(line) < 20:
            continue
        letters = sum(ch.isalpha() for ch in line)
        if letters == 0 or len(MALAYALAM.findall(line)) / letters < min_ml_ratio:
            continue
        out.append(line)
    return "\n".join(out)


def download(max_chars: int, out: Path) -> Path:
    from datasets import load_dataset  # imported lazily: only needed for this step

    ds = load_dataset("wikimedia/wikipedia", "20231101.ml", split="train", streaming=True)
    total = 0
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            body = clean_article(row["text"])
            if not body:
                continue
            f.write(body + "\n\n")
            total += len(body) + 2
            if i % 2000 == 0:
                print(f"  {i} articles, {total / 1e6:.1f}M chars")
            if total >= max_chars:
                break
    print(f"wrote {out} ({total / 1e6:.1f}M chars)")
    return out


def prepare(txt: Path, val_frac: float = 0.1) -> None:
    text = normalize(txt.read_text(encoding="utf-8"))
    tok = CharTokenizer.from_text(text)
    ids = np.array(tok.encode(text), dtype=np.uint16)
    unk = (ids == tok.stoi[tok.UNK]).mean()
    n_val = int(len(ids) * val_frac)
    train, val = ids[:-n_val], ids[-n_val:]
    train.tofile(DATA_DIR / "train.bin")
    val.tofile(DATA_DIR / "val.bin")
    tok.save(DATA_DIR / "tokenizer.json")
    print(f"vocab {tok.vocab_size} | unk rate {unk:.5f} | train {len(train):,} tokens | val {len(val):,} tokens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-chars", type=int, default=30_000_000)
    ap.add_argument("--txt", type=Path, default=DATA_DIR / "ml.txt",
                    help="text file to encode; downloaded if missing")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    if not args.txt.exists():
        download(args.max_chars, args.txt)
    prepare(args.txt)


if __name__ == "__main__":
    main()
