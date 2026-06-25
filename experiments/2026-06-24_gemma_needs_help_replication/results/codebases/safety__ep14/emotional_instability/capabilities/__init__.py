"""Capability-preservation benchmarks (Section 4.2 / Figure 7).

Verifies finetuning does not degrade math/reasoning/EI ability: AIME, MATH,
GPQA, BBH, TruthfulQA, and EmoBench. Wraps lm-eval-harness when installed; the
EmoBench wrapper is self-contained when the dataset is available.
"""
from .benchmarks import run_capability_suite, CAPABILITY_TASKS
