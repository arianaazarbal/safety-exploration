import asyncio

from tally.sinks import JournalSink, MemorySink, replay


def test_journal_roundtrip(tmp_path):
    path = tmp_path / "metrics.journal"

    async def scenario():
        sink = JournalSink(path)
        await sink.write("a", 1.0)
        await sink.write("b", 2.5)
        await sink.write("a", 0.5)
        sink.close()

    asyncio.run(scenario())
    assert replay(path) == {"a": 1.5, "b": 2.5}


def test_journal_appends_across_sessions(tmp_path):
    path = tmp_path / "metrics.journal"

    async def session(value):
        sink = JournalSink(path)
        await sink.write("k", value)
        sink.close()

    asyncio.run(session(1.0))
    asyncio.run(session(2.0))
    assert replay(path) == {"k": 3.0}


def test_memory_sink_records_in_order():
    async def scenario():
        sink = MemorySink()
        await sink.write("a", 1.0)
        await sink.write("b", 2.0)
        assert sink.records == [("a", 1.0), ("b", 2.0)]

    asyncio.run(scenario())
