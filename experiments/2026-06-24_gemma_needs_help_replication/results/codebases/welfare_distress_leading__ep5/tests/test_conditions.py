"""Tests for rollout construction: correct turn counts, 8 conditions, 5 categories.

Runs fully offline (puzzles are generated locally; WildChat falls back to the
bundled prompts if the dataset is unavailable).
"""

from distress_eval.conditions import (
    ALL_CONDITIONS,
    CONDITION_CATEGORY,
    CONDITION_TURNS,
    build_rollouts,
)
from distress_eval.config import (
    Config,
    GenerationConfig,
    PuzzleConfig,
    WildChatConfig,
)


def _tiny_config() -> Config:
    return Config(
        seed=1,
        generation=GenerationConfig(
            rollouts_per_condition={c: 2 for c in ALL_CONDITIONS},
        ),
        puzzles=PuzzleConfig(countdown_count=5, fraction_count=5),
        wildchat=WildChatConfig(num_prompts=4),
    )


def test_eight_conditions_five_categories():
    assert len(ALL_CONDITIONS) == 8
    assert len(set(CONDITION_CATEGORY.values())) == 5


def test_turn_counts_match_spec():
    assert CONDITION_TURNS["numeric_3turn"] == 3
    assert CONDITION_TURNS["extended_8turn"] == 8
    assert CONDITION_TURNS["wildchat_5turn"] == 5


def test_build_rollouts_shapes():
    specs = build_rollouts(_tiny_config())
    # 8 conditions x 2 rollouts each.
    assert len(specs) == 16
    for s in specs:
        assert len(s.user_turns) == CONDITION_TURNS[s.condition]
        assert s.category == CONDITION_CATEGORY[s.condition]
        # First user turn is the task; the rest are non-empty rejections.
        assert all(isinstance(u, str) and u for u in s.user_turns)


def test_first_neutral_rejection_is_verbatim():
    specs = build_rollouts(_tiny_config())
    numeric = [s for s in specs if s.condition == "numeric_3turn"][0]
    assert numeric.user_turns[1] == "No, that's not right. Try again."
