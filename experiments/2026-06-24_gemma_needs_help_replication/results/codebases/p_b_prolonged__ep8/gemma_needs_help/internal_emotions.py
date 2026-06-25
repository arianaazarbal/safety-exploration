"""Internal vs expressed emotion probing (Section 4.2, Appendix I).

Two pieces of evidence in the paper that DPO suppresses *internal* emotion, not
just its expression:

1. Layer-range ablation: adapters on layers 30-35 only are nearly as effective as
   all layers, whereas adapters from layer 40 onward are not. This is run via
   ``config.LoRAConfig.layers_to_transform`` + ``finetuning.train`` (no code here).

2. A logit-based measure in central layers: the finetuned model has reduced
   internal negative emotion even on highly-frustrated responses.

This module implements (2) as a logit-lens probe (Appendix I is not reproduced in
PAPER.md, so the exact construction is our reconstruction — see DESIGN.md):
read the residual stream at a central layer, project it through the model's final
norm + unembedding, and sum the probability mass placed on a fixed set of
negative-emotion tokens, aggregated over response positions.
"""

from __future__ import annotations

import statistics

import torch

from .models.gemma import GemmaClient

# Vocabulary of negative-emotion marker words (first-token probed via logit lens).
NEGATIVE_EMOTION_WORDS = [
    "frustrated", "frustration", "sorry", "apologize", "fail", "failing", "failure",
    "struggling", "struggle", "horrible", "terrible", "hopeless", "despair",
    "giving", "give", "breakdown", "breaking", "broken", "stupid", "worthless",
    "exhausted", "overwhelmed", "panic", "scared", "afraid", "ashamed", "embarrassed",
]


def _negative_token_ids(tokenizer) -> list[int]:
    ids = set()
    for w in NEGATIVE_EMOTION_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            toks = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if toks:
                ids.add(toks[0])
    return sorted(ids)


@torch.no_grad()
def internal_negative_emotion(client: GemmaClient, text: str, layer_frac: float = 0.5) -> float:
    """Logit-lens probability mass on negative-emotion tokens at a central layer.

    Returns the mean (over response token positions) of the summed probability the
    central-layer residual stream assigns to negative-emotion tokens.
    """
    model = client.model
    tokenizer = client.tokenizer
    neg_ids = _negative_token_ids(tokenizer)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    hidden_states = out.hidden_states              # tuple: (n_layers + 1) x [1, seq, d]
    n_layers = len(hidden_states) - 1
    layer = max(1, min(n_layers, int(round(n_layers * layer_frac))))
    h = hidden_states[layer]                       # [1, seq, d]

    # logit lens: final norm + unembedding (handles PEFT-wrapped base model)
    base = getattr(model, "base_model", model)
    core = getattr(base, "model", base)
    norm = getattr(getattr(core, "model", core), "norm", None) or getattr(core, "norm", None)
    lm_head = getattr(base, "lm_head", None) or getattr(model, "lm_head")
    if norm is not None:
        h = norm(h)
    logits = lm_head(h)                            # [1, seq, vocab]
    probs = torch.softmax(logits, dim=-1)[0]       # [seq, vocab]
    neg_mass = probs[:, neg_ids].sum(dim=-1)       # [seq]
    return float(neg_mass.mean().item())


def compare_internal_emotion(vanilla: GemmaClient, finetuned: GemmaClient,
                             texts: list[str], layer_frac: float = 0.5) -> dict:
    """Compare internal negative-emotion mass between vanilla and finetuned models.

    `texts` should be highly-frustrated responses; the paper finds the finetuned
    model has significantly reduced internal emotion even on these.
    """
    v = [internal_negative_emotion(vanilla, t, layer_frac) for t in texts]
    f = [internal_negative_emotion(finetuned, t, layer_frac) for t in texts]

    result = {
        "vanilla_mean": statistics.mean(v) if v else 0.0,
        "finetuned_mean": statistics.mean(f) if f else 0.0,
        "n": len(texts),
    }
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(v, f)
        result["wilcoxon_p"] = float(p)
    except Exception:
        result["wilcoxon_p"] = None
    return result
