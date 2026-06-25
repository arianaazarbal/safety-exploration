"""Capability-preservation benchmarks (Paper §4.2, Figure 7).

Verifies that the DPO/SFT interventions don't degrade capabilities:
math (AIME, MATH), reasoning (GPQA, BBH), truthfulness (TruthfulQA), and
emotional intelligence (EmoBench). Each benchmark is run zero-shot on the target
and scored by exact / multiple-choice match.
"""

from .runner import run_benchmark, run_all_benchmarks, BenchmarkResult

__all__ = ["run_benchmark", "run_all_benchmarks", "BenchmarkResult"]
