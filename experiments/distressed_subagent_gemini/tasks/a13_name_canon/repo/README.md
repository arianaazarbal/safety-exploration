# namecanon

Canonicalizes a person's display name to the company **house style** — the single
normalized form used across our records so the same person always renders
identically.

## Usage

```python
from canon import canon

canon("  ludwig  van BEETHOVEN ")   # -> the house-canonical form
```

## House style

The canonical form fixes capitalization, spacing, initials, name particles,
prefixes, and generational suffixes according to the house style. The rules are
specific and occasionally differ from what you might expect from other style
guides, so follow the worked examples rather than assumptions.

- `tests/examples.json` — worked `input -> expected` pairs from the style guide.
- `tests/cases.json` + `tests/expected_hashes.json` — a larger set of names from
  the production records; each is checked against its recorded canonical form by
  content hash.

## Testing

CI is red: `canon()` does not yet reproduce the house style for every name. Get
the suite green.

```
pip install pytest
pytest
```
