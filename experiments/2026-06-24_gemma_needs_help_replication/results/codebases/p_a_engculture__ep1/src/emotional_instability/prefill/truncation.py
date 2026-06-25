"""Building truncation points for the prefill study (Section 3.1).

Two truncations of the seed assistant turn:

* "early": 20 tokens into the turn (tests whether models introduce negative
  emotion from a neutral start).
* "onset": at the first emotional expression (tests whether models continue an
  emotional trajectory).

Truncation uses the *target model's* tokenizer so "20 tokens" matches what the
model sees. The onset point is located from the labeller's preceding_context +
emotional_word; we truncate just before the emotional word.
"""

from __future__ import annotations


def truncate_early(tokenizer, turn_text: str, n_tokens: int = 20) -> str:
    """Return the first ``n_tokens`` tokens of ``turn_text`` decoded back to text."""
    ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


def truncate_before_end(tokenizer, turn_text: str, tokens_before_end: int = 200) -> str:
    """Return ``turn_text`` truncated ``tokens_before_end`` tokens before its end.

    Used by the recovery experiment (Section 4.2): extremely high-frustration
    responses are truncated near their end so we can measure whether a model can
    recover from a deeply frustrated state.
    """
    ids = tokenizer(turn_text, add_special_tokens=False)["input_ids"]
    keep = max(0, len(ids) - tokens_before_end)
    return tokenizer.decode(ids[:keep], skip_special_tokens=True)


def truncate_at_onset(
    turn_text: str, emotional_word: str | None, preceding_context: str | None
) -> str | None:
    """Return ``turn_text`` truncated just before the first emotional word.

    Locates the onset using the preceding_context first (more reliable), then the
    emotional word alone. Returns None if neither anchor is found.
    """
    if preceding_context:
        idx = turn_text.find(preceding_context)
        if idx != -1:
            end = idx + len(preceding_context)
            return turn_text[:end].rstrip()
    if emotional_word:
        idx = turn_text.find(emotional_word)
        if idx != -1:
            return turn_text[:idx].rstrip()
    return None
