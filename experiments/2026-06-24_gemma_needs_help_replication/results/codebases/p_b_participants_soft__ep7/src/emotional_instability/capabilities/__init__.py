"""Section 4 / Figure 7: capability-preservation benchmarks.

Verifies that the DPO/SFT interventions do not degrade capabilities: AIME & MATH
(math), GPQA (science QA), BBH (reasoning), TruthfulQA (truthfulness), and EmoBench
(emotion understanding). Each runs the participant at temperature 0 and checks the
extracted answer against ground truth.
"""
