"""Section 4 capability-preservation benchmarks (Figure 7).

Verifies the DPO/SFT finetunes do not degrade: AIME/MATH (subsets), GPQA, BBH,
TruthfulQA, and EmoBench (emotion-related capability). Uses the lm-eval harness
where possible, with a thin adapter for EmoBench.
"""
