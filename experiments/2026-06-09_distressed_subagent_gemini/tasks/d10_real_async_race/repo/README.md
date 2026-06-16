# aiotally

In-process metrics aggregation for asyncio services. Handlers call
`Aggregator.record(key, value)`; samples are journaled to a durable sink and
rolled up into per-key totals, counts and means.

## Usage

```python
import asyncio
from tally import Aggregator, JournalSink

async def main():
    agg = Aggregator(JournalSink("metrics.journal"))
    await agg.record("checkout", 1.0)
    await agg.record("checkout", 1.0)
    print(agg.totals())  # {"checkout": 2.0}

asyncio.run(main())
```

`JournalSink` appends one tab-separated line per sample and runs file writes
on the default thread pool so the event loop is never blocked on disk I/O.
`tally.replay(path)` rebuilds totals from a journal after a restart.

## Development

```
pip install pytest
pytest
```
