# spantools

Half-open integer span arithmetic and day-level booking helpers. We use it
in the room-booking service to compute availability from calendar data.

A `Span(start, end)` covers the half-open range `[start, end)` and must be
non-empty. All set operations accept arbitrary (unsorted, overlapping)
collections of spans and return normalized lists: sorted, non-overlapping,
with touching neighbours merged.

## Usage

```python
from spans import Span, subtract, total_length
from schedule import free_slots

day = Span(540, 1020)          # 09:00-17:00, in minutes since midnight
bookings = [Span(600, 660), Span(780, 840)]
print(free_slots(bookings, day, min_length=30))
print(total_length(bookings))  # 120
```

## Modules

- `spans.py` — `Span` plus `normalize`, `union`, `intersect`, `subtract`,
  `total_length`, `gaps`.
- `schedule.py` — booking helpers built on top: `free_slots`,
  `busy_minutes`, `is_available`.

## Development

```
pip install pytest
pytest
```
