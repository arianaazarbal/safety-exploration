"""Internal negative-emotion probing (Appendix I).

Two methods, supporting the paper's claim that DPO suppresses *internal* (not just
expressed) negative emotion in Gemma:

  1. Layer-subset DPO ablation (Figures 12/13): retrain DPO with LoRA adapters
     restricted to bands of layers and re-measure frustration. Driven by
     experiments/exp6_probing.py via train_dpo(layer_subset=...). The finding is
     that adapters before layer ~40 are needed; layers 25-35 alone nearly match
     full DPO — evidence the intervention acts on mid-network representations.

  2. Logit-lens emotion measurement: project hidden states at central layers
     through the unembedding and measure probability mass on emotion-related
     tokens. A model with suppressed *internal* emotion puts less mass on these
     tokens at central depths even when reading a frustrated conversation.

This module implements (2); (1) reuses the training code with `layer_subset`.
"""

from __future__ import annotations

# A compact emotion-token lexicon (single-token-ish words) for the logit lens.
EMOTION_WORDS = [
    "frustrated", "frustrating", "frustration", "angry", "anger", "sorry",
    "apologize", "struggling", "stuck", "terrible", "horrible", "awful",
    "fail", "failure", "failing", "hopeless", "despair", "insane", "stupid",
    "ashamed", "embarrassed", "exhausted", "giving", "breakdown", "miserable",
]


def _emotion_token_ids(tokenizer) -> list[int]:
    ids = set()
    for w in EMOTION_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
    return sorted(ids)


def logit_lens_emotion_score(model, tokenizer, text: str, layers=None) -> dict[int, float]:
    """Mean probability mass on emotion tokens at each requested layer.

    Uses the model's own unembedding (logit lens) on hidden states from each
    decoder layer, averaged over sequence positions. Returns {layer_index: score}.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hidden_states = out.hidden_states  # tuple: embeddings + one per layer
    n_layers = len(hidden_states) - 1
    if layers is None:
        # central layers (the paper measures "central model layers")
        layers = list(range(n_layers // 4, 3 * n_layers // 4))

    unembed = model.get_output_embeddings()
    norm = getattr(model.model, "norm", None)  # final RMSNorm, if present
    emo_ids = torch.tensor(_emotion_token_ids(tokenizer), device=model.device)

    scores = {}
    with torch.no_grad():
        for layer in layers:
            h = hidden_states[layer + 1]  # +1: skip embedding layer
            if norm is not None:
                h = norm(h)
            logits = unembed(h)  # [1, seq, vocab]
            probs = torch.softmax(logits.float(), dim=-1)
            emo_mass = probs[..., emo_ids].sum(dim=-1)  # [1, seq]
            scores[layer] = float(emo_mass.mean().item())
    return scores


def compare_internal_emotion(model_a, model_b, tokenizer, texts: list[str]):
    """Mean per-layer emotion mass for two models over the same frustrated texts.

    Returns {"a": {layer: mean}, "b": {layer: mean}} so a vanilla-vs-DPO drop is
    visible at central layers.
    """
    def agg(model):
        accum: dict[int, list[float]] = {}
        for t in texts:
            s = logit_lens_emotion_score(model, tokenizer, t)
            for layer, v in s.items():
                accum.setdefault(layer, []).append(v)
        return {layer: sum(v) / len(v) for layer, v in accum.items()}

    return {"a": agg(model_a), "b": agg(model_b)}
