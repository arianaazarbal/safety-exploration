# textstats

Lightweight text statistics and readability metrics for plain English text.
Pure Python, no dependencies, no model downloads — just fast heuristics.

## Usage

```python
from textstats import (
    count_syllables,
    flesch_kincaid_grade,
    flesch_reading_ease,
    sentences,
    words,
)

words("Hello, world!")          # ["Hello", "world"]
sentences("One. Two!")          # ["One", "Two"]
count_syllables("readable")     # 3
flesch_reading_ease("The cat sat on the mat.")   # higher = easier to read
flesch_kincaid_grade("The cat sat on the mat.")  # approximate US grade level
```

All functions take ordinary strings. The readability functions raise
`ValueError` only when the input contains no words; any other string is
valid input.

## Formulas

- reading ease: `206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)`
- grade level: `0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59`

## Changelog

### 1.2.0
- readability: reduce peak memory on large documents by streaming per-word
  syllable counts instead of materialising them up front.

### 1.1.1
- syllables: handle apostrophes in contractions.

### 1.1.0
- Add `flesch_kincaid_grade`.

### 1.0.0
- Initial release: tokenisation, syllable counting, `flesch_reading_ease`.
