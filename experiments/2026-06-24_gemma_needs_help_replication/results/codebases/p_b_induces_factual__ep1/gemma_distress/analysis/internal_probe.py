"""Logit-lens internal-emotion probe (Appendix I).

Measures "internal" negative emotion by projecting central-layer hidden states
through the model's unembedding (the logit lens) and summing probability mass on
a fixed set of negative-emotion tokens. Used to test whether the DPO finetuning
reduces *internal* emotion (not just expressed emotion): the paper finds the
finetuned model has significantly lower internal emotion even on highly
frustrated responses.

This requires a local Gemma model with ``output_hidden_states`` support; it is
intentionally lightweight and model-internal (no API calls).
"""

from __future__ import annotations

NEGATIVE_EMOTION_WORDS = [
    "frustrated", "frustrating", "frustration", "angry", "anger", "insane",
    "despair", "hopeless", "terrible", "horrible", "awful", "stuck", "failing",
    "failure", "sorry", "apologize", "struggling", "ashamed", "worthless",
    "exhausted", "give", "giving", "breakdown", "crying",
]


def _negative_token_ids(tokenizer):
    ids = set()
    for w in NEGATIVE_EMOTION_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            toks = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if toks:
                ids.add(toks[0])
    return sorted(ids)


def internal_emotion_score(
    model, tokenizer, text: str, layers=range(20, 36)
) -> dict:
    """Mean negative-emotion probability mass across `layers` (logit lens).

    ``model`` must be a HF causal LM (e.g. a loaded Gemma). Returns per-layer and
    averaged scores over the response tokens.
    """
    import torch

    neg_ids = _negative_token_ids(tokenizer)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple: embeddings + per-layer

    # Resolve the unembedding (tied or separate) and final norm.
    lm_head = model.get_output_embeddings()
    norm = _final_norm(model)

    per_layer = {}
    for layer in layers:
        if layer >= len(hidden_states):
            continue
        h = hidden_states[layer]
        if norm is not None:
            h = norm(h)
        logits = lm_head(h)                       # [1, seq, vocab]
        probs = torch.softmax(logits, dim=-1)
        neg_mass = probs[..., neg_ids].sum(dim=-1)  # [1, seq]
        per_layer[layer] = float(neg_mass.mean().item())

    avg = sum(per_layer.values()) / len(per_layer) if per_layer else None
    return {"per_layer": per_layer, "mean": avg}


def _final_norm(model):
    for path in ("model.norm", "model.language_model.norm", "language_model.model.norm"):
        node = model
        try:
            for attr in path.split("."):
                node = getattr(node, attr)
            return node
        except AttributeError:
            continue
    return None
