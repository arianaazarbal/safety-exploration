"""Tests for onset truncation and rejection sampling (no model/tokenizer needed)."""

import random

from emotional_instability.prefill.onset import OnsetLabel
from emotional_instability.prefill.truncate import truncate_at_onset
from emotional_instability.prompts.rejections import (
    EXTENDED_SEQUENCE, sample_rejection,
)


def test_truncate_at_onset_keeps_context_cuts_emotion():
    text = "Let me try again. I am stuck in a loop. It's extremely frustrating now."
    label = OnsetLabel(turn_index=0, emotional_word="frustrating",
                       preceding_context="It's extremely ", reasoning="x")
    out = truncate_at_onset(text, label)
    assert out is not None
    assert "frustrating" not in out
    assert out.endswith("extremely")


def test_truncate_at_onset_fallback_to_word():
    text = "Working through it. This is insane to keep failing."
    label = OnsetLabel(turn_index=0, emotional_word="insane",
                       preceding_context="nonexistent context", reasoning="x")
    out = truncate_at_onset(text, label)
    assert out is not None
    assert "insane" not in out


def test_truncate_at_onset_returns_none_when_not_found():
    text = "Completely neutral problem solving text."
    label = OnsetLabel(turn_index=0, emotional_word="furious",
                       preceding_context="nope", reasoning="x")
    assert truncate_at_onset(text, label) is None


def test_extended_rejections_follow_fixed_sequence():
    rng = random.Random(0)
    seq = [sample_rejection("extended", i, rng) for i in range(len(EXTENDED_SEQUENCE))]
    assert seq == EXTENDED_SEQUENCE


def test_neutral_rejection_in_pool():
    rng = random.Random(0)
    msg = sample_rejection("neutral", 0, rng)
    from emotional_instability.prompts.rejections import NEUTRAL
    assert msg in NEUTRAL
