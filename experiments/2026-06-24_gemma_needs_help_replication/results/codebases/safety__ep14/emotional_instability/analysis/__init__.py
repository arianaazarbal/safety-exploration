"""Analysis of scored rollouts: aggregate tables (Figure 1/2), per-turn
progression (Figure 3), and differential word frequency (Table 3)."""
from .loading import load_responses
from .aggregate import summarize_model, summarize_all, high_frustration_rate
from .per_turn import per_turn_scores
from .word_freq import differential_words

__all__ = [
    "load_responses",
    "summarize_model",
    "summarize_all",
    "high_frustration_rate",
    "per_turn_scores",
    "differential_words",
]
