"""Section 3: comparing base and instruct models via prefilling.

Pipeline:
  1. seeds.py      - select high-frustration seed conversations from a Gemma-it run
  2. onset.py      - locate emotion onset in each seed (Claude, Appendix C.1)
  3. truncate.py   - build "early" (20-token) and "onset" truncations
  4. paraphrase.py - paraphrase truncations to remove Gemma stylistic bias (C.2)
  5. runner.py     - generate 50 continuations/prefill for base & instruct, judge them
"""
