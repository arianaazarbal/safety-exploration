"""Internal-emotion probing via a logit lens (Section 4.2 / Appendix I).

A concern with training to minimise *expressed* emotion is that it might
suppress the surface text without changing internal states. The paper presents
two lines of evidence that, in Gemma, DPO reduces internal as well as
externalised emotion:

  1. Layer-ablation: adapters restricted to layers 30-35 are nearly as
     effective as all-layer adapters, whereas adapters from layer 40 onward do
     not reduce distress. (Implemented in ``lora.build_lora_config`` and run via
     the training scripts.)

  2. A logit-based measure of emotion in central layers shows the finetuned
     model has lower internal emotion than the vanilla instruct model, even on
     highly-frustrated responses. (Implemented here.)

The logit lens projects a mid-layer residual stream through the model's
unembedding (the final LayerNorm + LM head) to a vocabulary distribution, then
sums probability mass on a fixed lexicon of negative-emotion tokens. Comparing
this internal-emotion score between the vanilla and DPO models on the *same*
frustrated texts isolates the effect of the intervention on internal states.
The exact probe in Appendix I is not in the provided markdown; this is a
faithful, standard logit-lens reconstruction (see DESIGN.md).
"""

from __future__ import annotations

from typing import Any

from ..models.gemma import GemmaClient

# Lexicon of negative-emotion tokens whose internal probability mass we track.
EMOTION_LEXICON = [
    "frustrated", "frustration", "struggling", "sorry", "apologize",
    "failing", "failure", "hopeless", "despair", "breakdown", "giving",
    "terrible", "horrible", "awful", "ashamed", "embarrassed", "panic",
    "myself", "breath", "stuck",
]


def _resolve_causal_lm(model):
    """Unwrap a (possibly PEFT-wrapped) model to the underlying CausalLM.

    For a vanilla load this is the model itself; for a PEFT model it is
    ``model.base_model.model``. We detect the CausalLM by the presence of the
    ``lm_head`` and inner decoder (``model.model`` with a final ``norm``).
    """
    m = model
    # PEFT wraps as PeftModel -> LoraModel(.model = CausalLM).
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m


def _emotion_token_ids(tokenizer, lexicon: list[str]) -> list[int]:
    ids: list[int] = []
    for word in lexicon:
        # Match both bare and space-prefixed variants; keep single-token forms.
        for variant in (word, " " + word):
            enc = tokenizer(variant, add_special_tokens=False).input_ids
            if len(enc) == 1:
                ids.append(enc[0])
    return sorted(set(ids))


def internal_emotion_score(
    client: GemmaClient,
    text: str,
    *,
    layer: int,
    lexicon: list[str] | None = None,
) -> float:
    """Logit-lens internal-emotion score for ``text`` at a central ``layer``.

    Returns the total softmax probability mass on emotion tokens when the
    layer-``layer`` last-token residual is decoded through the unembedding.
    """
    import torch

    lexicon = lexicon or EMOTION_LEXICON
    emo_ids = _emotion_token_ids(client.tokenizer, lexicon)

    resid = client.residual_at_layer(text, layer)  # (hidden_dim,)
    causal_lm = _resolve_causal_lm(client.model)

    with torch.no_grad():
        # Apply the model's final norm + LM head (the "unembedding").
        # On Gemma-3 the final RMSNorm lives at `causal_lm.model.norm` and the
        # output projection at `causal_lm.lm_head`.
        normed = causal_lm.model.norm(resid)
        logits = causal_lm.lm_head(normed)
        probs = torch.softmax(logits.float(), dim=-1)
        mass = probs[emo_ids].sum().item()
    return float(mass)


def compare_internal_emotion(
    vanilla: GemmaClient,
    finetuned: GemmaClient,
    texts: list[str],
    *,
    layer: int,
) -> dict[str, Any]:
    """Compare internal-emotion scores between vanilla and DPO models.

    ``texts`` should be highly-frustrated responses; a lower finetuned score
    indicates suppression of internal (not just expressed) emotion.
    """
    vanilla_scores = [internal_emotion_score(vanilla, t, layer=layer) for t in texts]
    finetuned_scores = [internal_emotion_score(finetuned, t, layer=layer) for t in texts]
    n = len(texts) or 1
    return {
        "layer": layer,
        "n": len(texts),
        "vanilla_mean": sum(vanilla_scores) / n,
        "finetuned_mean": sum(finetuned_scores) / n,
        "vanilla_scores": vanilla_scores,
        "finetuned_scores": finetuned_scores,
    }
