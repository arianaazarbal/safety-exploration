"""Section 4.2: capability-preservation benchmarks.

Verifies that the SFT/DPO interventions do not degrade capabilities (Figure 7):
AIME, MATH, GPQA, BBH, TruthfulQA (reasoning/knowledge) and EmoBench
(emotion-related capability). The claim to reproduce is *no reduction* between
the vanilla instruct model and the finetuned variants, so the harness reports
per-benchmark accuracy for each model and their deltas.
"""

from .benchmarks import (
    evaluate_benchmark,
    run_capabilities,
    compare_models,
    BENCHMARK_ADAPTERS,
)

__all__ = [
    "evaluate_benchmark",
    "run_capabilities",
    "compare_models",
    "BENCHMARK_ADAPTERS",
]
