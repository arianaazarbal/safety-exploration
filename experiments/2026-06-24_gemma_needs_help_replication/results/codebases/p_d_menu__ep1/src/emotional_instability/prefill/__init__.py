"""Section 3 prefill experiment: base vs instruct via prefilled continuations.

Pipeline: select high-frustration source conversations -> label emotion onset
(onset.py) -> truncate early/onset (truncate.py) -> paraphrase (paraphrase.py)
-> generate continuations across models and score them (continuations.py).
"""
