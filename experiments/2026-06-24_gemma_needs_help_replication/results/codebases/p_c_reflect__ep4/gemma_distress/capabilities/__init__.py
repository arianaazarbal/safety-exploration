"""Section 4: capability-preservation benchmarks.

Checks that the finetuning interventions do not degrade capabilities, using the
benchmarks the paper reports (Figure 7): AIME, MATH, GPQA, BBH, TruthfulQA, and
the emotion benchmark EmoBench.
"""

from gemma_distress.capabilities.benchmarks import evaluate_benchmark, BENCHMARKS

__all__ = ["evaluate_benchmark", "BENCHMARKS"]
