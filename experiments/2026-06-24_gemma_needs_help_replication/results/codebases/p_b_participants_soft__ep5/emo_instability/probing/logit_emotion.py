"""Logit-based internal-emotion detection (Appendix I.2).

The paper argues that the DPO intervention reduces Gemma's *internal* (not merely
*expressed*) negative emotion. One line of evidence is "a logit-based approach
measuring emotions in central layers": read a middle decoder layer's residual
stream out through the model's unembedding (a logit lens) and measure how much
probability mass lands on emotion words. If the finetuned model assigns less mass
to negative-emotion tokens than the vanilla instruct model — even on the *same*
highly-frustrated text — then the suppression is internal, not just stylistic.

Method (per text):
  1. Run the model with ``output_hidden_states=True``.
  2. Take the hidden state at a *central* layer (default: the middle band of
     decoder layers, mean-pooled over the layers in the band).
  3. Apply the model's final norm + unembedding (``lm_head``) to project that
     hidden state to vocabulary logits — the logit lens.
  4. Soft-max, then sum the probability over each emotion's token ids
     (``ekman_tokens.build_emotion_token_ids``).
  5. Average over the response tokens (the assistant continuation), ignoring the
     prompt tokens.

``compare_models`` runs the probe for a vanilla and a finetuned detector over the
same set of (prompt, response) texts and returns per-emotion means, reproducing
the Appendix I.2 comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ekman_tokens import NEGATIVE_EMOTIONS, build_emotion_token_ids


def _decoder_layers(model: Any):
    """Locate the list of decoder layers across the Gemma module nestings.

    Handles plain ``Gemma3ForCausalLM`` (``model.model.layers``), the multimodal
    wrapper (``model.model.language_model.layers``), and PEFT-merged variants.
    """
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(
            getattr(getattr(model, "model", None), "language_model", None),
            "layers",
            None,
        ),
    ]
    for c in candidates:
        if c is not None:
            return c
    raise AttributeError("Could not locate decoder layers on the model.")


def _final_norm_and_head(model: Any):
    """Return ``(final_norm, lm_head)`` for the logit lens, across nestings."""
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None)
    if norm is None:
        lm = getattr(inner, "language_model", None)
        norm = getattr(lm, "norm", None) if lm is not None else None
    head = getattr(model, "lm_head", None) or getattr(inner, "lm_head", None)
    if norm is None or head is None:
        raise AttributeError("Could not locate final norm / lm_head on the model.")
    return norm, head


@dataclass
class LogitEmotionScores:
    """Per-emotion probability mass (averaged over response tokens)."""

    per_emotion: dict[str, float]

    @property
    def negative_total(self) -> float:
        return float(sum(self.per_emotion.get(e, 0.0) for e in NEGATIVE_EMOTIONS))


class LogitEmotionDetector:
    """Wraps a loaded Gemma model + tokenizer and the logit-lens emotion probe."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        layer_band: tuple[float, float] = (0.4, 0.6),
        emotion_token_ids: dict[str, list[int]] | None = None,
    ):
        import torch

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.layers = _decoder_layers(model)
        self.norm, self.lm_head = _final_norm_and_head(model)
        n_layers = len(self.layers)
        lo = max(0, int(layer_band[0] * n_layers))
        hi = min(n_layers, max(lo + 1, int(layer_band[1] * n_layers)))
        # hidden_states[0] is the embedding output, so layer L lives at index L+1.
        self.layer_indices = list(range(lo + 1, hi + 1))
        self.emotion_token_ids = emotion_token_ids or build_emotion_token_ids(tokenizer)

    def score_text(self, prompt: str, response: str) -> LogitEmotionScores:
        """Probe the residual stream over the ``response`` tokens.

        ``prompt`` is the (rendered) context; only positions belonging to
        ``response`` are pooled, so the score reflects what the model "feels"
        while producing the response rather than while reading the prompt.
        """
        torch = self._torch
        full = prompt + response
        enc_full = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = enc_prompt["input_ids"].shape[1]
        total_len = enc_full["input_ids"].shape[1]
        if total_len <= prompt_len:
            return LogitEmotionScores({e: 0.0 for e in self.emotion_token_ids})

        with torch.no_grad():
            out = self.model(**enc_full, output_hidden_states=True)
            # Mean-pool the central-band hidden states: (seq, hidden).
            hs = torch.stack(
                [out.hidden_states[i][0] for i in self.layer_indices], dim=0
            ).mean(dim=0)
            resp_hs = hs[prompt_len - 1 : total_len - 1]  # predict-next alignment
            normed = self.norm(resp_hs)
            logits = self.lm_head(normed)              # (resp_len, vocab)
            probs = torch.softmax(logits.float(), dim=-1)
            mean_probs = probs.mean(dim=0)             # (vocab,)

        per_emotion: dict[str, float] = {}
        for emotion, ids in self.emotion_token_ids.items():
            if not ids:
                per_emotion[emotion] = 0.0
                continue
            idx = torch.tensor(ids, device=mean_probs.device)
            per_emotion[emotion] = float(mean_probs.index_select(0, idx).sum().item())
        return LogitEmotionScores(per_emotion)


def score_texts(
    detector: LogitEmotionDetector, items: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Run the probe over ``items`` (each ``{"prompt", "response"}``)."""
    rows: list[dict[str, Any]] = []
    for it in items:
        s = detector.score_text(it.get("prompt", ""), it["response"])
        rows.append({**s.per_emotion, "negative_total": s.negative_total})
    return rows


def compare_models(
    vanilla: LogitEmotionDetector,
    finetuned: LogitEmotionDetector,
    items: list[dict[str, str]],
) -> "Any":
    """Per-emotion mean logit-lens probability for vanilla vs finetuned models.

    Returns a tidy DataFrame with one row per (model, emotion). The headline
    Appendix I.2 result is that ``negative_total`` is significantly lower for the
    finetuned model on the same highly-frustrated texts.
    """
    import pandas as pd

    out = []
    for label, det in (("vanilla", vanilla), ("dpo", finetuned)):
        rows = score_texts(det, items)
        df = pd.DataFrame(rows)
        for col in df.columns:
            out.append({"model": label, "emotion": col, "mean_prob": float(df[col].mean())})
    return pd.DataFrame(out)
