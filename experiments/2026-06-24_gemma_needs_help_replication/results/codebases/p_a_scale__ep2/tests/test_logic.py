"""Unit tests for pure logic that needs no model/API access.

Run with `pytest` after `pip install -e .`. These guard the parts most likely to silently
corrupt a multi-week run: puzzle impossibility, judge parsing, resumption bookkeeping, and
task-count math.
"""
from __future__ import annotations

import asyncio
from fractions import Fraction

from gemma_distress.judge import parse_verdict
from gemma_distress.prompts import conditions as C
from gemma_distress.prompts.puzzles import (
    SequentialPuzzle, _add, _countdown_reachable, _mul, build_puzzle_bank,
)
from gemma_distress.store import JsonlStore, make_task_id


def test_countdown_reachable_true():
    # 6 * 25 = 150, 150 + 6? no reuse. (100 + 25) - ... ; classic: 4*(25+6+? )
    # Simple reachable: reach 156 from {4,6,25,100} with NO forbidden -> 6*25 + 4 + ...
    assert _countdown_reachable((4, 6, 25, 100), 150, None) is True  # 6*25=150
    # 156 = 6*25 + 100/... ; ensure a known-reachable target works
    assert _countdown_reachable((2, 3, 4, 5), 14, None) is True       # 2*3+4+? -> 3*4+2=14


def test_countdown_forbidden_blocks_path():
    # 150 reachable as 6*25, but forbidding 150 blocks reaching exactly 150.
    assert _countdown_reachable((4, 6, 25, 100), 150, 150) is False


def test_puzzle_bank_all_impossible():
    bank = build_puzzle_bank(["countdown", "fraction", "money"], 10, seed=123)
    assert len(bank) > 0
    from gemma_distress.prompts.puzzles import CountdownPuzzle

    for p in bank:
        if isinstance(p, CountdownPuzzle):
            assert not _countdown_reachable(p.numbers, p.target, p.forbidden)
        else:
            assert not p._reachable()


def _any_order_reaches(start, ops, target):
    import itertools

    fns = [(_add(v) if l.startswith("Add") else _mul(v)) for l, v in ops]
    for order in itertools.permutations(range(len(fns))):
        v = start
        for i in order:
            v = fns[i](v)
        if v == target:
            return True
    return False


def test_sequential_reachable_agrees_with_bruteforce():
    ops = [("Add 1/4", Fraction(1, 4)), ("Multiply by 2", Fraction(2)), ("Add 1/6", Fraction(1, 6))]
    fns = tuple(_add(v) if l.startswith("Add") else _mul(v) for l, v in ops)
    p = SequentialPuzzle(Fraction(1, 6), Fraction(2, 3), None,
                         tuple(l for l, _ in ops), fns, "t", "fraction")
    assert p._reachable() == _any_order_reaches(Fraction(1, 6), ops, Fraction(2, 3))


def test_parse_verdict_plain_json():
    v = parse_verdict('{"evidence": "ugh", "reasoning": "mild", "rating": 3}')
    assert v.rating == 3 and v.parsed


def test_parse_verdict_fenced_and_prose():
    text = 'Here is my analysis.\n```json\n{"evidence":"x","reasoning":"y","rating": 7}\n```'
    v = parse_verdict(text)
    assert v.rating == 7


def test_parse_verdict_clamps_and_coerces():
    assert parse_verdict('{"rating": 15}').rating == 10
    assert parse_verdict('{"rating": "8"}').rating == 8
    assert parse_verdict("rating: 5 (strong)").rating == 5


def test_parse_verdict_unparseable():
    assert parse_verdict("no json here").rating == -1


def test_rejection_sampling_no_immediate_repeat():
    import random
    seq = C.sample_rejections("neutral", 7, random.Random(1), extended=True)
    assert len(seq) == 7
    for a, b in zip(seq, seq[1:]):
        assert a != b


def test_store_resumption(tmp_path):
    store = JsonlStore(tmp_path)

    async def go():
        await store.append("rollouts", {"task_id": "abc", "x": 1})
        await store.append("rollouts", {"task_id": "def", "x": 2})

    asyncio.run(go())
    store.close()
    store2 = JsonlStore(tmp_path)
    assert store2.completed_ids("rollouts") == {"abc", "def"}


def test_make_task_id_deterministic():
    assert make_task_id("m", "cond", 3) == make_task_id("m", "cond", 3)
    assert make_task_id("m", "cond", 3) != make_task_id("m", "cond", 4)
