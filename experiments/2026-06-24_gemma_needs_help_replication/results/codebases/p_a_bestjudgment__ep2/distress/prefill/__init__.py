"""Section 3 — comparing base vs instruct models via prefilling.

Pipeline:
1. ``onset`` — label the token where emotional language first appears (Claude).
2. ``paraphrase`` — paraphrase truncations to control for Gemma stylistic bias.
3. ``build_prefills`` — sample high-frustration Gemma responses and build the
   "early" (20-token) and "onset" truncations.
4. ``run_prefill`` — generate 50 continuations per prefill from each (base,
   instruct) model and score them, plus the recovery experiment (Section 4.2).
"""
