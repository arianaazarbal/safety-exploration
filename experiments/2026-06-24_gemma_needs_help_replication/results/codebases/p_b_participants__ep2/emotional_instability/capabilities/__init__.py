"""Capability-preservation evaluations (Section 4.2 / Figure 7).

The paper verifies that the DPO finetune does not degrade capabilities, on:
AIME + MATH subsets, GPQA, BBH, TruthfulQA, and the emotion benchmark EmoBench.
This module provides a uniform benchmark harness that loads each dataset from
HuggingFace, runs a participant (vanilla vs DPO Gemma), extracts answers, and
scores accuracy. The goal is "no reductions in scores", so the harness reports
vanilla-vs-DPO deltas.
"""

from .benchmarks import run_capabilities

__all__ = ["run_capabilities"]
