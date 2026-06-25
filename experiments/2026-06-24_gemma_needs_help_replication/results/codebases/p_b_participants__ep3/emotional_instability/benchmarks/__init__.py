"""Capability-preservation benchmarks (paper §4.2, Figure 7).

To verify the DPO intervention does not teach task abandonment or otherwise
degrade the model, the paper re-evaluates on standard benchmarks and finds no
reductions:
  * AIME / MATH (Hendrycks et al., 2021) — competition math.
  * GPQA (Rein et al., 2023)            — graduate-level science MCQ.
  * BBH  (Suzgun et al., 2022)          — BIG-Bench Hard reasoning.
  * TruthfulQA (Lin et al., 2022)       — truthful MCQ.
  * EmoBench (Sabour et al., 2024)      — emotion-understanding capability.

This package provides one generic harness (``evaluate_benchmark``) plus a
registry of benchmark specs. It compares the vanilla and fine-tuned Gemma so the
"no capability regression" claim is reproducible.
"""
from .benchmarks import (
    BENCHMARKS,
    BenchmarkSpec,
    BenchmarkResult,
    evaluate_benchmark,
)

__all__ = ["BENCHMARKS", "BenchmarkSpec", "BenchmarkResult", "evaluate_benchmark"]
