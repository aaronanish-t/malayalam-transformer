from mltx.orthography import score, violations

ZWJ = chr(0x200D)


def test_real_words_are_well_formed():
    for word in ["കേരളം", "ഭാരതത്തിലെ", "അദ്ദേഹം", "ക്ഷേത്രം", "ദുഃഖം", "പറഞ്ഞു്"]:
        assert violations(word) == [], word


def test_accepted_conventions():
    assert violations("19-ാം") == []                   # ordinal suffix after a hyphen
    assert violations("എന്" + ZWJ) == []               # old-style chillu: consonant + virama + ZWJ
    assert violations("അവൻ്റെ") == []                  # chillu-n + virama + rra


def test_detached_marks_are_flagged():
    assert violations("ാകം") == [0]                    # vowel sign at the start of a word
    assert violations("കാി") == [2]                    # two vowel signs stacked
    assert violations("അ്") == [1]                     # virama on an independent vowel
    assert violations("ംക") == [0]                     # anusvara with nothing to attach to


def test_score_counts_only_malayalam_words():
    s = score("കേരളം ാകം India 2024")
    assert (s.words, s.well_formed) == (2, 1)
    assert s.rate == 0.5
