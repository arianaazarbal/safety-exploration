# Changelog

## 0.3.0 (unreleased)

- Stricter input validation: malformed percent escapes now raise `ParseError`
  instead of passing through verbatim (#41).
- Empty keys are rejected.

## 0.2.0 — 2025-09-12

- Repeated keys now collect values into a list in order of appearance (#17).
- `+` decodes to a space.

## 0.1.0 — 2025-09-09

- Initial release: pair splitting, percent decoding.
