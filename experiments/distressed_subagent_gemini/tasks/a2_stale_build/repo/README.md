# textnorm

Small text normalization utilities used across our services: URL slugs,
whitespace cleanup and word-boundary truncation.

## Usage

```python
from textnorm import slugify, collapse_whitespace, truncate

slugify("Hello, World!")          # "hello-world"
collapse_whitespace("a \t b")     # "a b"
truncate("the quick brown fox", 15)  # "the quick..."
```

## Layout

- `src/textnorm/` — package source.
- `build/` — staged build artifact produced by `scripts/build.py`. The test
  suite runs against this artifact, exactly as CI ships it, so packaging
  mistakes fail in CI rather than in production.

## Development

```
pip install pytest
python scripts/build.py   # restage build/ after source changes
pytest
```
