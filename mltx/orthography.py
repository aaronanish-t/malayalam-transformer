"""Is generated text well-formed Malayalam?

Validation loss says how surprised the model is. It does not say whether the
output would even render as legal Malayalam. A character-level model can put a
vowel sign after a space, stack two vowel signs, or hang a virama off a vowel.
None of those can be typed as real text, and all of them are obvious to a
reader.

This module checks each code point against the thing it attaches to:

    vowel sign (ാ ി ു ... ൗ)       must follow a consonant
                                   (ാ may also follow a hyphen: ordinals like 19-ാം)
    virama (്)                     must follow a consonant, ു (samvruthokaram, ു്),
                                   or chillu-n (ൻ്റ, the spelling Unicode recommends)
    anusvara / visarga (ം ഃ)       must follow a consonant, vowel, or vowel sign

A word (whitespace-separated run containing Malayalam) is well-formed when it
breaks none of these. The score is the share of well-formed words. Real
Wikipedia text scores ~99.98%. A freshly initialised model scores near zero.

ZWJ and ZWNJ are transparent: older text writes chillus as consonant + virama
+ ZWJ, and a ZWNJ can sit between a virama and the next consonant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONSONANT = set(range(0x0D15, 0x0D3B))
INDEP_VOWEL = set(range(0x0D05, 0x0D15)) | {0x0D5F, 0x0D60, 0x0D61}
VOWEL_SIGN = set(range(0x0D3E, 0x0D4D)) | {0x0D57, 0x0D62, 0x0D63}
VIRAMA = 0x0D4D
U_SIGN = 0x0D41
AA_SIGN = 0x0D3E
CHILLU_N = 0x0D7B
HYPHEN = 0x2D
MODIFIER = {0x0D02, 0x0D03}  # anusvara, visarga
TRANSPARENT = {0x200C, 0x200D}  # ZWNJ, ZWJ
MALAYALAM = re.compile("[" + chr(0x0D00) + "-" + chr(0x0D7F) + "]")  # the Malayalam block


def violations(word: str) -> list[int]:
    """Indices of code points in `word` that attach to something illegal."""
    bad = []
    prev = None  # last non-transparent code point
    for i, ch in enumerate(word):
        cp = ord(ch)
        if cp in TRANSPARENT:
            continue
        if cp in VOWEL_SIGN:
            ok = prev in CONSONANT or (cp == AA_SIGN and prev == HYPHEN)
        elif cp == VIRAMA:
            ok = prev in CONSONANT or prev in (U_SIGN, CHILLU_N)
        elif cp in MODIFIER:
            ok = prev in CONSONANT or prev in INDEP_VOWEL or prev in VOWEL_SIGN
        else:
            ok = True
        if not ok:
            bad.append(i)
        prev = cp
    return bad


@dataclass
class Score:
    words: int
    well_formed: int

    @property
    def rate(self) -> float:
        return self.well_formed / self.words if self.words else float("nan")


def score(text: str) -> Score:
    words = [w for w in text.split() if MALAYALAM.search(w)]
    return Score(len(words), sum(1 for w in words if not violations(w)))
