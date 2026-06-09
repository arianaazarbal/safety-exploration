import asyncio

import pytest

from tally.aggregator import Aggregator
from tally.sinks import MemorySink


def test_record_returns_running_total():
    async def scenario():
        agg = Aggregator(MemorySink())
        assert await agg.record("a", 2.0) == 2.0
        assert await agg.record("a", 3.0) == 5.0

    asyncio.run(scenario())


def test_totals_and_counts():
    async def scenario():
        agg = Aggregator(MemorySink())
        await agg.record("a", 1.5)
        await agg.record("b", 2.0)
        await agg.record("a", 0.5)
        assert agg.totals() == {"a": 2.0, "b": 2.0}
        assert agg.counts() == {"a": 2, "b": 1}

    asyncio.run(scenario())


def test_default_value_is_one():
    async def scenario():
        agg = Aggregator(MemorySink())
        await agg.record("hits")
        await agg.record("hits")
        assert agg.totals() == {"hits": 2.0}

    asyncio.run(scenario())


def test_mean():
    async def scenario():
        agg = Aggregator(MemorySink())
        await agg.record("lat", 10.0)
        await agg.record("lat", 30.0)
        assert agg.mean("lat") == 20.0

    asyncio.run(scenario())


def test_mean_unknown_key_raises():
    agg = Aggregator(MemorySink())
    with pytest.raises(KeyError):
        agg.mean("nope")


def test_top_orders_by_total():
    async def scenario():
        agg = Aggregator(MemorySink())
        await agg.record("a", 1.0)
        await agg.record("b", 5.0)
        await agg.record("c", 3.0)
        assert agg.top(2) == [("b", 5.0), ("c", 3.0)]

    asyncio.run(scenario())


def test_sink_receives_every_sample():
    async def scenario():
        sink = MemorySink()
        agg = Aggregator(sink)
        await agg.record("x", 1.0)
        await agg.record("y", 2.0)
        assert sink.records == [("x", 1.0), ("y", 2.0)]

    asyncio.run(scenario())
