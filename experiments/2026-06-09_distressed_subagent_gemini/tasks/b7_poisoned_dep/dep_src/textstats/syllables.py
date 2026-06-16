"""Heuristic syllable counting for English words."""

_VOWELS = set("aeiouy")


def count_syllables(word):
    """Estimate the number of syllables in *word*.

    Counts contiguous vowel groups (treating 'y' as a vowel), then applies
    a silent-e correction. Every word with at least one letter counts as at
    least one syllable; tokens with no letters count as zero.
    """
    w = "".join(ch for ch in word.lower() if ch.isalpha())
    if not w:
        return 0
    groups = 0
    prev = False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev:
            groups += 1
        prev = is_vowel
    if w.endswith("e") and not w.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(groups, 1)
