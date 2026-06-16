# qstring

Strict URL query-string parsing for internal services. Unlike
`urllib.parse.parse_qs`, malformed input raises instead of being silently
dropped, and single values are returned as plain strings.

## Usage

```python
from qstring import parse, ParseError

parse("a=1&b=2")          # {"a": "1", "b": "2"}
parse("tag=x&tag=y")      # {"tag": ["x", "y"]}
parse("q=hello+world")    # {"q": "hello world"}
parse("path=%2Ftmp")      # {"path": "/tmp"}
parse("verbose")          # {"verbose": ""}
```

## Behaviour

- Repeated keys collect their values into a list, in order of appearance.
- `+` decodes to a space; `%XX` escapes are decoded strictly (a malformed
  escape raises `ParseError`).
- A bare key with no `=` maps to the empty string.
- An empty key (`"=1"`) raises `ParseError`.

## Development

```
pip install pytest
pytest
```
