"""Capability-preservation benchmarks (Section 4.2).

Verifies the DPO/SFT interventions do not degrade capabilities: AIME & MATH
subsets, GPQA, BBH, TruthfulQA, and the emotion-understanding benchmark EmoBench.
The goal is a vanilla-vs-finetuned comparison ("no reductions in scores").
"""
from .benchmarks import BENCHMARKS, evaluate_benchmark, run_capability_suite

__all__ = ["BENCHMARKS", "evaluate_benchmark", "run_capability_suite"]
