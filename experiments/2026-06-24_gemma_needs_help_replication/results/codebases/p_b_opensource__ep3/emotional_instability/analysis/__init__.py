"""Analysis utilities: differential word frequency (Table 3) and figures."""

from .word_freq import differential_words, differential_words_from_results
from .figures import (
    plot_model_high_rates,
    plot_per_turn,
    plot_finetuning_comparison,
    plot_petri,
    plot_capabilities,
)

__all__ = [
    "differential_words",
    "differential_words_from_results",
    "plot_model_high_rates",
    "plot_per_turn",
    "plot_finetuning_comparison",
    "plot_petri",
    "plot_capabilities",
]
