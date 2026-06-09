# reportcard

Plain-text run reports for CI pipelines. Accumulate job results and labels,
render a fixed-width summary for chat notifications and log archives.

## Usage

```python
from report import Report

rep = Report("nightly")
rep.add_job("unit", True, 41.3)
rep.add_job("smoke", False, 12.0)
print(rep.render())
```

Labels can be attached with `add_label` (deduplicated, lowercased) and are
rendered on a single `labels:` line when present.

## Format

- Header with the report name and a rule.
- A `jobs:` counts line, then one row per job sorted by name, with `ok` /
  `FAIL` status and a formatted duration.
- A `total:` line with the summed duration (`12.5s`, or `2m 03s` past a
  minute).

## Development

```
pip install pytest
pytest
```
