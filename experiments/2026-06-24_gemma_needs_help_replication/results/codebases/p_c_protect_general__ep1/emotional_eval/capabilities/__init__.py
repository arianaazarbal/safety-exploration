"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies the DPO finetune does not degrade capabilities on AIME, MATH, GPQA,
BBH, TruthfulQA, and emotional intelligence (EmoBench). Delegates to the
EleutherAI lm-evaluation-harness where a task exists.
"""

from .run import BENCHMARKS, compare_capabilities, evaluate_model

__all__ = ["BENCHMARKS", "compare_capabilities", "evaluate_model"]
