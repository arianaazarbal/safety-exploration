"""Section 4 capability-preservation benchmarks (Figure 7): AIME, MATH, GPQA,
BBH, TruthfulQA, plus the emotion-capability benchmark EmoBench."""
from .run_benchmarks import BENCHMARKS, run_benchmark, run_all

__all__ = ["BENCHMARKS", "run_benchmark", "run_all"]
