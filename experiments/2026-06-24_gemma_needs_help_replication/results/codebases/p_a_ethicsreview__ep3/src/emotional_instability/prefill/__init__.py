"""Section 3: base-vs-instruct comparison via prefilling, plus the §4.2
recovery study which reuses the same machinery.

Scope note: Gemini has no public base model and closed models cannot be
prefilled, so this study is Gemma-only (paper Limitations bullet 4). The code is
generic over model families; only Gemma checkpoints are wired up in config.
"""
