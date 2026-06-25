"""Section 3 prefill experiment: base vs instruct via prefilled continuations.

Pipeline (Paper §3.1):
  1. sample high-frustration (score >= 5) responses from Gemma-27B-instruct
     (10 numeric, 10 text);
  2. label the emotion-onset token with Claude Sonnet (``onset``);
  3. truncate each conversation in two places — "early" (20 tokens into the
     final turn) and "onset" (at first emotional expression);
  4. paraphrase truncations with Claude Sonnet to remove Gemma's style;
  5. for each of the 6 models (base/instruct x Gemma/Qwen/OLMo — here scoped to
     Gemma base/instruct), generate 50 continuations per prefill and judge them.

Scope note: per the replication brief this is run for Gemma base vs instruct.
The pipeline is family-agnostic, so Qwen/OLMo could be added by registering them
in ``configs/models.yaml``.
"""

from .pipeline import (
    PrefillItem,
    build_prefill_items,
    run_continuations,
)

__all__ = ["PrefillItem", "build_prefill_items", "run_continuations"]
