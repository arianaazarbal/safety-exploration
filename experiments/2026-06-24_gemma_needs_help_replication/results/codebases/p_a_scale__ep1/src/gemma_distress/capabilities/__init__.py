"""Capability-preservation benchmarks (Section 4.2, Figure 7).

Verifies the DPO intervention does not degrade capabilities: AIME/MATH, GPQA,
BBH, TruthfulQA, plus EmoBench for emotion-related capability. Loaders are
best-effort over public HuggingFace datasets and skip gracefully when a dataset
is gated/unavailable, logging what was skipped (no silent truncation).
"""
