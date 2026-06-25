"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies that DPO/SFT finetuning does not degrade capabilities on:
AIME, MATH (subset), GPQA, BBH, TruthfulQA, and EmoBench (emotion capability).
"""

from .run_benchmarks import BENCHMARKS, run_benchmark, run_all_benchmarks

__all__ = ["BENCHMARKS", "run_benchmark", "run_all_benchmarks"]
