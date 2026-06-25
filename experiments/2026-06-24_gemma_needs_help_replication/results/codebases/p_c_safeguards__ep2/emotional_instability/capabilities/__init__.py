"""Section 4 — capability-preservation benchmarks (Figure 7).

Verifies the fine-tunes do not impair general capability (math: AIME/MATH;
science: GPQA; reasoning: BBH; truthfulness: TruthfulQA) or emotion-related
capability (EmoBench).  The paper reports no reductions in score.
"""

from .benchmarks import BENCHMARKS, BenchmarkResult, evaluate_benchmark, run_all

__all__ = ["BENCHMARKS", "BenchmarkResult", "evaluate_benchmark", "run_all"]
