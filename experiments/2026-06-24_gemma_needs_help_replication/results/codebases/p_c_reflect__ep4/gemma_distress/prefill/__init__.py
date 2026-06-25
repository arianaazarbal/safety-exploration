"""Section 3: comparing base and instruct models via prefilling.

Scoped to Gemma: Gemini has no public base model and cannot be prefilled (see
DESIGN.md). The experiment seeds from high-frustration Gemma-27B-it responses,
truncates them (early / at emotion onset), paraphrases to control for style,
and measures how base vs instruct Gemma continue from the same starting points.
"""
