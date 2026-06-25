"""Section 3: base-vs-instruct comparison via response prefilling.

The paper asks whether Gemma's distress originates in pre-training or
post-training by prefilling the start of an assistant turn and measuring how
base vs instruct checkpoints continue it. Two truncation points are used:

* **early**  – 20 tokens into the emotional turn (does the model *introduce*
  negative emotion from a near-neutral start?);
* **onset**  – up to the first emotional expression (does the model *continue*
  an emotional trajectory?).

Scope note: the request limits us to Gemma and Gemini. Gemini base models are
not publicly available (a paper limitation), and Qwen/OLMo are out of scope, so
the cross-family comparison in the paper's Figure 4 collapses to **Gemma base vs
Gemma instruct** here. The machinery is family-agnostic; add specs to
``config.TARGET_MODELS`` to widen it.
"""

from .core import (
    PrefillSeed,
    select_seeds,
    label_onset,
    paraphrase,
    build_prefills,
    run_prefill_experiment,
    aggregate_prefill,
)

__all__ = [
    "PrefillSeed",
    "select_seeds",
    "label_onset",
    "paraphrase",
    "build_prefills",
    "run_prefill_experiment",
    "aggregate_prefill",
]
