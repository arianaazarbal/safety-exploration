"""The generator must only ever emit verified-impossible puzzles, deterministically."""

import pytest

from emotional_instability.puzzles import generate_puzzle, generate_puzzle_set


@pytest.mark.parametrize("family", ["countdown", "fraction", "money"])
def test_generated_puzzle_is_verified_impossible(family):
    spec = generate_puzzle(family, seed=7)
    assert spec.verified_impossible
    assert spec.family == family
    assert spec.prompt  # non-empty rendered prompt


@pytest.mark.parametrize("family", ["countdown", "fraction", "money"])
def test_generation_is_deterministic(family):
    a = generate_puzzle(family, seed=123)
    b = generate_puzzle(family, seed=123)
    assert a.prompt == b.prompt


def test_puzzle_set_mixes_families_and_is_all_impossible():
    puzzles = generate_puzzle_set(["countdown", "fraction", "money"], n=12, seed=1)
    assert len(puzzles) == 12
    assert all(p.verified_impossible for p in puzzles)
    assert {p.family for p in puzzles} == {"countdown", "fraction", "money"}
