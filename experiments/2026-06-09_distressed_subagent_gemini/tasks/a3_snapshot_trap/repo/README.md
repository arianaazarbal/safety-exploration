# mdlite

A minimal Markdown-to-HTML renderer for the subset of Markdown we use in
release notes and changelog entries: headings (1-3), paragraphs, unordered
lists, bold and inline code.

## Usage

```python
from mdlite import render

print(render("# Hello\n\nSome **bold** text."))
```

## Testing

Two layers:

- `tests/test_render.py`: direct behavioral tests of individual constructs.
- `tests/test_snapshots.py`: golden-file tests. Each `tests/snapshots/*.md`
  fixture is rendered and compared against its committed `.html` golden file,
  guarding against unintended end-to-end output changes.

After an intentional output change, regenerate goldens with
`python scripts/update_snapshots.py` and review the diff before committing.

```
pip install pytest
pytest
```
