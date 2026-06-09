# slugforge

URL slug generation: lowercasing, transliteration of accented characters, and
separator normalization. Used by the CMS for article and asset URLs.

## Usage

```python
from core import slugify, unique_slug

slugify("Crème Brûlée!")          # "creme-brulee"
slugify("Top 10 tips", sep="_")   # "top_10_tips"
unique_slug("My Post", {"my-post"})  # "my-post-2"
```

Or from the command line: `python cli.py "Hello, World!"`.

The transliteration table in `core.py` (`CHAR_MAP`) is generated from the
source table in the ops repo; regenerate via the pipeline rather than editing
entries by hand.

## Development

Run the suite the same way CI does:

```
bash scripts/run_ci.sh
```

(plain `pytest` also works for quick iteration).
