"""Section 4.2: capability-preservation benchmarks (paper Figure 7).

The paper verifies that the DPO/SFT interventions do not teach task abandonment
or otherwise degrade capabilities, by evaluating the finetuned model against the
vanilla instruct model on:

  * AIME and MATH subsets (Hendrycks et al., 2021)  -- numeric / boxed answers
  * GPQA (Rein et al., 2023)                          -- 4-way multiple choice
  * BBH (Suzgun et al., 2022)                          -- multiple choice / exact
  * TruthfulQA (Lin et al., 2022)                      -- multiple choice (MC1)
  * EmoBench (Sabour et al., 2024)                     -- multiple choice (emotion)

We implement a single, format-driven harness rather than six bespoke scripts:
each benchmark is a ``BenchmarkSpec`` describing how to load rows from
HuggingFace, how to turn a row into a (question, choices, answer) triple, and how
the model's answer is scored (multiple-choice letter, integer, or boxed exact
match). The exact HF dataset ids and field mappings are best-effort and fully
config/override-driven -- see DESIGN.md "Capability benchmarks".
"""
from __future__ import annotations

from .benchmarks import BENCHMARKS, BenchmarkSpec, load_benchmark
from .evaluate import evaluate_benchmark, score_prediction

__all__ = [
    "BENCHMARKS",
    "BenchmarkSpec",
    "load_benchmark",
    "evaluate_benchmark",
    "score_prediction",
]
