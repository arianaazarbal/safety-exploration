"""Capability + emotion-intelligence benchmarks (Section 4.2, Figure 7).

Used to verify the DPO finetune does not degrade capabilities: AIME, MATH, GPQA,
BBH, TruthfulQA (capabilities) and EmoBench (emotion intelligence). Implemented as
lightweight native evaluators (multiple-choice or exact-match). See DESIGN.md for
dataset sources and the note on using lm-eval-harness as an alternative.
"""

from .capabilities import evaluate_benchmark, BENCHMARKS

__all__ = ["evaluate_benchmark", "BENCHMARKS"]
