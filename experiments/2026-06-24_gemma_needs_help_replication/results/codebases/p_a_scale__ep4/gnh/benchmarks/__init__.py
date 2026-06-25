"""Capability + emotion-intelligence benchmarks (Section 4.2 / Fig 7).

These verify the DPO finetune does not degrade capabilities. We implement a
compact, resumable harness covering free-form math (AIME, MATH) and
multiple-choice tasks (GPQA, BBH, TruthfulQA, EmoBench). Dataset schemas vary by
source/version, so the adapters in `suites.py` are best-effort and log when they
cannot map a row; see DESIGN.md.
"""
