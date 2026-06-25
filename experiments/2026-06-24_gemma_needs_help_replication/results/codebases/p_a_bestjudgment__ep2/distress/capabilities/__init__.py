"""Section 4.2 — capability preservation (Figure 7).

Evaluate AIME, MATH, GPQA, BBH, TruthfulQA and EmoBench to verify the DPO/SFT
interventions do not degrade capabilities. A generic harness normalises each
benchmark into a common ``{question, choices, answer}`` shape, generates with
the target model, extracts an answer, and computes accuracy.
"""

from .run_capabilities import evaluate_benchmarks, BenchmarkResult  # noqa: F401
