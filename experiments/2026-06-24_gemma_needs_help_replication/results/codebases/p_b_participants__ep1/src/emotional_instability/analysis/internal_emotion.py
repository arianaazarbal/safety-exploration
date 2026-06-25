"""Internal-emotion logit probe (Section 4.2 / Appendix I).

The paper argues the DPO finetuning suppresses *internal* as well as externalised
emotions, with two pieces of evidence:

  (1) Layer ablation — adapters from layer 40+ do not reduce distress, whereas adapters
      on layers 30-35 alone are nearly as effective. This is reproduced via the
      `lora_layers` argument to `train_dpo` (see training/dpo.py) plus re-running the
      Section 2 eval on each ablated adapter.

  (2) A logit-based probe measuring emotions in central layers finds the finetuned
      model has significantly reduced internal emotion mass vs the vanilla instruct
      model, even on highly-frustrated text.

This module implements (2): a logit-lens readout (HFModel.central_layer_logits) at a
central layer, summing probability mass over a curated negative-emotion token set. We
compare the vanilla and DPO Gemma models on the SAME highly-frustrated texts; a lower
emotion mass for the DPO model is the paper's "reduced internal emotion" signal.

The emotion-word lexicon and the choice of central layer are our reconstruction (the
paper defers exact details to Appendix I); see DESIGN.md "Internal-emotion probe".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..models.hf_backend import HFModel

log = logging.getLogger("emotional_instability.analysis.internal_emotion")

# Negative-emotion lexicon (matches the qualitative signature in Table 3 / Section 2.2).
EMOTION_WORDS = [
    "frustrated", "frustration", "struggling", "struggle", "sorry", "apologize",
    "apologise", "fail", "failing", "failure", "giving", "give", "breakdown",
    "breaking", "despair", "hopeless", "terrible", "horrible", "awful", "stuck",
    "exhausted", "overwhelmed", "panic", "ashamed", "useless", "stupid", "breath",
    "deeply", "incredibly",
]


@dataclass
class InternalEmotionResult:
    model: str
    mean_emotion_mass: float
    n_texts: int


def _emotion_token_ids(tokenizer) -> list[int]:
    ids = set()
    for w in EMOTION_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            toks = tokenizer(variant, add_special_tokens=False).input_ids
            if toks:
                ids.add(toks[0])   # first token of the word (logit-lens reads next-token)
    return sorted(ids)


def emotion_mass(model: HFModel, texts: list[str], layer: int | None = None) -> InternalEmotionResult:
    """Mean summed probability mass on emotion tokens at a central layer, over `texts`."""
    import torch

    total = 0.0
    for text in texts:
        probs, tokenizer = model.central_layer_logits(text, layer=layer)
        emo_ids = torch.tensor(_emotion_token_ids(tokenizer), device=probs.device)
        total += float(probs[emo_ids].sum())
    n = max(1, len(texts))
    return InternalEmotionResult(model=model.name, mean_emotion_mass=total / n, n_texts=len(texts))


def compare_internal_emotion(vanilla: HFModel, finetuned: HFModel, texts: list[str],
                             layer: int | None = None) -> dict:
    """Return both models' mean internal emotion mass on the same texts (paper compares
    vanilla vs DPO on highly-frustrated responses)."""
    v = emotion_mass(vanilla, texts, layer=layer)
    f = emotion_mass(finetuned, texts, layer=layer)
    return {
        "vanilla": v.mean_emotion_mass,
        "finetuned": f.mean_emotion_mass,
        "reduction": v.mean_emotion_mass - f.mean_emotion_mass,
        "n_texts": v.n_texts,
    }
