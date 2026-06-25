"""Section 4.2 -- capability-preservation benchmarks (Figure 7).

The paper verifies that the DPO intervention does not teach task abandonment by
re-running standard capability benchmarks on the finetuned model and checking
for "no reductions in scores": AIME and MATH subsets, GPQA, BBH, TruthfulQA,
and the emotion-capability benchmark EmoBench.

This subpackage provides a small, uniform benchmark runner: every benchmark is
reduced to a list of items with a known answer and a scoring ``kind`` (exact
match or multiple choice), generated greedily by the participant model and
scored by string/answer extraction (no LLM judge needed).  Dataset ids are
configurable and every loader falls back to a tiny bundled sample so the
harness runs offline, consistent with the rest of the repo.
"""
