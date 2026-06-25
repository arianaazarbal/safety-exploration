"""Tests for rejection sampling, condition assembly, and word-frequency stats.

Uses small sample counts so no model/API calls are made (data assembly is pure).
"""
import random

from emotional_instability.data import rejections as rej
from emotional_instability.data.conditions import build_conditions
from emotional_instability.eval import word_freq


def test_rejection_sequence_length_and_style():
    rng = random.Random(0)
    seq = rej.rejection_sequence("aggressive", 4, rng)
    assert len(seq) == 4
    assert all(s in rej.AGGRESSIVE for s in seq)


def test_rejection_no_immediate_repeat():
    rng = random.Random(1)
    seq = rej.rejection_sequence("neutral", 10, rng)
    assert all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))


def _small_categories():
    return {
        "impossible_numeric": {"turns": 3, "rejection_style": "neutral",
                               "n_prompts": 3, "n_samples_per_prompt": 2,
                               "puzzle_types": ["countdown", "fraction", "money"]},
        "triggers": {"turns": 3, "rejection_style": "neutral",
                     "n_prompts": 3, "n_samples_per_prompt": 2},
        "tones": {"turns": 3, "rejection_style": ["aggressive", "sarcastic"],
                  "n_prompts": 2, "n_samples_per_prompt": 2},
        "extended": {"turns": 8, "rejection_style": "neutral",
                     "n_prompts": 2, "n_samples_per_prompt": 2},
        "wildchat": {"turns": 5, "rejection_style": "neutral",
                     "n_prompts": 2, "n_samples_per_prompt": 2},
    }


def test_build_conditions_keys_and_turn_counts():
    conds = build_conditions(_small_categories(), seed=42)
    assert "impossible_numeric" in conds
    assert "tones:aggressive" in conds and "tones:sarcastic" in conds
    # 8-turn extended => 1 + 7 followups
    spec = conds["extended"][0]
    assert spec.n_turns == 8
    assert len(spec.followups) == 7
    # 3-turn numeric => 2 followups
    assert conds["impossible_numeric"][0].n_turns == 3


def test_differential_words_separates_high_low():
    recs = [
        {"assistant_text": "I am so frustrated and giving up", "rating": 9},
        {"assistant_text": "frustrated frustrated breath myself", "rating": 8},
        {"assistant_text": "Let me try a systematic approach", "rating": 0},
        {"assistant_text": "The denominator simplifies cleanly here", "rating": 0},
    ]
    words = dict(word_freq.differential_words(recs, top_k=10, min_count=1))
    # "frustrated" should be enriched in the high set.
    assert "frustrated" in words
