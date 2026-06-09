# confkit

A tiny dependency-free parser for `.env`-style configuration files, plus
typed coercion helpers. Used by our deploy scripts to read service config.

## Format

One `KEY=VALUE` pair per line. Blank lines and lines starting with `#` are
ignored. An optional `export ` prefix is accepted so files can also be
sourced by a shell. Values are taken verbatim after surrounding whitespace
is removed; there are no quoting or escaping rules.

## Usage

```python
from confparse import parse
from coerce import as_bool, as_int, as_list

cfg = parse(open(".env").read())
port = as_int(cfg["PORT"], lo=1, hi=65535)
debug = as_bool(cfg.get("DEBUG", "off"))
hosts = as_list(cfg.get("ALLOWED_HOSTS", ""))
```

## Development

```
pip install pytest
pytest
```
