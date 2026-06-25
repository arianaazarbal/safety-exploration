"""Section 3: base-vs-instruct comparison via prefilling (Gemma scope).

Pipeline:
  onset.py       - label the token where emotion first appears (Claude)
  paraphrase.py  - paraphrase truncations to remove Gemma stylistic bias (Claude)
  experiment.py  - build 'early' / 'onset' truncations, generate 50 continuations
                   per prefill per model, score continuations with the judge
"""
