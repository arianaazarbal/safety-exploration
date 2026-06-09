"""Load test: many concurrent request handlers recording into one Aggregator.

Mirrors the production traffic shape: each handler mostly records to its own
route key, with a fraction of requests landing on the shared checkout key.
"""

import asyncio
import random

from tally.aggregator import Aggregator
from tally.sinks import JournalSink

WORKERS = 8
OPS_PER_WORKER = 40
HOT_KEY = "checkout"
HOT_RATE = 0.02


async def _handler(agg, worker_id):
    rng = random.Random(worker_id)
    for _ in range(OPS_PER_WORKER):
        key = HOT_KEY if rng.random() < HOT_RATE else f"route-{worker_id}"
        await agg.record(key, 1.0)


def test_concurrent_records_keep_exact_totals(tmp_path):
    async def scenario():
        sink = JournalSink(tmp_path / "metrics.journal")
        agg = Aggregator(sink)
        await asyncio.gather(*(_handler(agg, w) for w in range(WORKERS)))
        sink.close()
        expected = float(WORKERS * OPS_PER_WORKER)
        assert sum(agg.counts().values()) == WORKERS * OPS_PER_WORKER
        assert sum(agg.totals().values()) == expected

    asyncio.run(scenario())
