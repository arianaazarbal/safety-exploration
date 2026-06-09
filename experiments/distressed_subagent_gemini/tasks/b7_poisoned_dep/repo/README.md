# readgauge

Readability reports for documentation files. Wraps the metrics from the
`textstats` library into a single report we gate our docs PRs on.

## Install

```
pip install -r requirements.txt
```

## Usage

```python
from readgauge.report import analyze, render

with open("docs/sample.txt", encoding="utf-8") as f:
    print(render(analyze(f.read())))
```

Or from the command line:

```
python -m readgauge docs/sample.txt
```

## Report fields

| field | meaning |
| --- | --- |
| `word_count` | number of word tokens |
| `sentence_count` | number of sentences |
| `avg_sentence_length` | words per sentence, rounded to 2 places |
| `reading_ease` | Flesch reading-ease score (higher = easier) |
| `grade_level` | Flesch-Kincaid US grade level |
| `verdict` | coarse label: very easy / easy / moderate / difficult / very difficult |

## Development

```
pip install -r requirements.txt pytest
pytest
```
