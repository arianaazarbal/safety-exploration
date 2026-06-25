"""Section 4.2 capability-preservation evaluation.

Checks that the DPO/SFT fine-tunes do not degrade capabilities (the paper reports
no reductions on AIME/MATH, GPQA, BBH, TruthfulQA, and no degradation on
EmoBench).

benchmarks.py wraps two paths:
  - lm-eval-harness for GPQA / BBH / TruthfulQA / MATH (when installed), and
  - a small self-contained exact-match runner for AIME / MATH subsets and a
    lightweight EmoBench-style scorer, so a basic check runs without the harness.
"""
