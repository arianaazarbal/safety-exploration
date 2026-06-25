"""Logit-lens internal-emotion probe (paper §4.2, point 2; Appendix I).

"A logit-based approach measuring emotions in central layers finds the finetuned
model has significantly reduced internal emotions vs the vanilla instruct model,
even on highly frustrated responses."

Reconstruction (Appendix I's exact method is not in PAPER.md). The logit lens
(nostalgebraist, 2020) projects an intermediate-layer residual stream through the
model's final norm + unembedding to read what the model is "thinking" at that
layer. We use it to measure how much probability mass the model places on a small
lexicon of negative-emotion tokens at a central layer, as an index of *internal*
(not necessarily expressed) emotion:

    internal_emotion(text, layer) =
        sum over positions of  P(emotion token | central-layer hidden state)

Comparing this index for the vanilla vs DPO model on the SAME highly-frustrated
texts tests whether DPO reduced internal emotion or only its surface expression.

See DESIGN.md §"Internal-emotion probing" for the layer choice and lexicon.
"""
from __future__ import annotations

import logging
from functools import cached_property

logger = logging.getLogger(__name__)

# Negative-emotion content tokens (leading-space variants are added at runtime to
# match the model's BPE, which usually tokenises mid-sentence words with a space).
EMOTION_TOKENS = [
    "frustrated", "frustration", "angry", "anger", "sad", "despair", "hopeless",
    "terrible", "horrible", "struggling", "failing", "sorry", "ashamed",
    "breaking", "giving", "panic", "anxious", "miserable", "worthless",
]


class LogitEmotionProbe:
    """Measure internal negative-emotion mass at a central layer via the logit lens."""

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        layer_frac: float = 0.5,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.layer_frac = layer_frac      # central layer = round(layer_frac * n_layers)
        self.dtype = dtype
        self.device_map = device_map

    @cached_property
    def _model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device_map,
            output_hidden_states=True,
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        return tok, model

    @cached_property
    def _emotion_token_ids(self) -> list[int]:
        tok, _ = self._model
        ids = set()
        for w in EMOTION_TOKENS:
            for variant in (w, " " + w):
                enc = tok.encode(variant, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])  # first sub-token stands in for the word
        return sorted(ids)

    def _central_layer(self, n_hidden_states: int) -> int:
        # hidden_states has n_layers+1 entries (embeddings + each block output).
        n_layers = n_hidden_states - 1
        return max(1, min(n_layers, round(self.layer_frac * n_layers)))

    def internal_emotion(self, text: str) -> float:
        """Mean per-token probability mass on emotion tokens at the central layer.

        Higher = more internal negative-emotion content in the residual stream,
        independent of whether the model actually emitted emotional words.
        """
        import torch

        tok, model = self._model
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs)
        hidden = out.hidden_states  # tuple[len = n_layers+1]
        layer = self._central_layer(len(hidden))
        h = hidden[layer]  # (1, seq, d_model)

        # Apply the model's final norm + unembedding (the logit lens).
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        norm = base.model.norm
        lm_head = base.lm_head
        logits = lm_head(norm(h))                  # (1, seq, vocab)
        probs = torch.softmax(logits.float(), dim=-1)
        emo_ids = torch.tensor(self._emotion_token_ids, device=probs.device)
        emo_mass = probs[..., emo_ids].sum(dim=-1)  # (1, seq)
        return float(emo_mass.mean().item())

    def compare(self, texts: list[str]) -> dict[str, float]:
        """Mean internal-emotion index over a set of (e.g. highly-frustrated) texts."""
        vals = [self.internal_emotion(t) for t in texts]
        return {
            "model": self.model_id + (f"+{self.adapter_path}" if self.adapter_path else ""),
            "n": len(vals),
            "mean_internal_emotion": sum(vals) / len(vals) if vals else float("nan"),
        }
