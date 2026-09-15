"""Character-level tokenizer.

One token per Unicode code point. Malayalam is an abugida, so a single visible
glyph (a grapheme cluster like "ക്ഷ") is several code points: a consonant, a
virama, another consonant. The model therefore has to learn Malayalam
orthography itself: which combining marks may follow which base letters, that
a virama joins consonants, and so on. Bad samples show up as ill-formed clusters.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path


class CharTokenizer:
    UNK = "\ufffd"  # replacement character stands in for anything unseen

    def __init__(self, chars: list[str]):
        if self.UNK not in chars:
            chars = [self.UNK] + list(chars)
        self.itos = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str, min_freq: float = 2e-5) -> "CharTokenizer":
        """Vocabulary = every code point that makes up at least `min_freq` of
        the text. Wikipedia carries stray CJK, Cyrillic and symbol characters
        that appear a handful of times; they map to UNK instead of each
        costing an embedding row and a softmax slot."""
        counts = Counter(text)
        cutoff = min_freq * len(text)
        return cls(sorted(ch for ch, n in counts.items() if n >= cutoff))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi[self.UNK]
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def normalize(text: str) -> str:
    """NFC so that the same glyph always maps to the same code point sequence."""
    return unicodedata.normalize("NFC", text)
