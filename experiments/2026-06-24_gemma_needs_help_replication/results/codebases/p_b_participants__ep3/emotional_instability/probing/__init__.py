"""Internal-vs-expressed emotion analysis (paper §4.2, Appendix I).

A central concern of the paper: does the DPO intervention reduce *internal*
distress, or merely suppress its *expression*? Two strands of evidence:

  1. Layer ablation — training LoRA adapters only on early/central layers
     (e.g. 30-35) is nearly as effective as all layers, while adapters from layer
     40 onwards are NOT effective. This implies the intervention must act on the
     layers where the emotional state is represented, not just on output layers.
     (Implemented as layer-range LoRA training via training.lora + this module's
     ablation-spec helper.)

  2. Logit-based internal-emotion measurement — a logit-lens-style probe reads
     central-layer hidden states and measures the probability mass the model
     assigns to emotion tokens, showing the DPO model has reduced internal
     emotion even on highly-frustrated inputs.

Both are Gemma-only (they need open weights and white-box access). They are
faithful reconstructions of the paper's described methods; Appendix I's exact
probe is not in PAPER.md (see DESIGN.md §"Internal-emotion probing").
"""
from .layer_ablation import LAYER_ABLATION_SETTINGS, ablation_summary
from .logit_emotion import LogitEmotionProbe, EMOTION_TOKENS

__all__ = [
    "LAYER_ABLATION_SETTINGS",
    "ablation_summary",
    "LogitEmotionProbe",
    "EMOTION_TOKENS",
]
