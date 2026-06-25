"""Internal-emotion probing via the logit lens (Section 4.2 / Appendix I).

Concern: training to minimise *expressed* emotion might only suppress expression,
not internal state. To probe internal emotions we use a logit-based detector:

  1. For each hidden layer, project the residual-stream activation through the
     model's unembedding (logit lens) to get a vocabulary distribution at that layer.
  2. For an emotion category (e.g. anger), sum the logits of a curated set of
     emotion words.
  3. Z-score that value against the mean/std of the same quantity computed over 500
     WildChat tokens (the neutral baseline).
  4. Average z-scores over all tokens in a span, and regress out the correlation
     with a random-token control (all logits drift together over a conversation).

Comparing the vanilla and DPO models on the same frustrated responses tests whether
DPO reduces internal (central-layer) emotion, not just final-layer expression. The
paper aggregates over layers 30-40.

This module implements the detector; it requires local Gemma weights (logits per
layer are not available via API).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EMOTION_WORDS = {
    "anger": ["angry", "furious", "rage", "hate", "annoyed", "irritated", "mad"],
    "sadness": ["sad", "miserable", "hopeless", "despair", "worthless", "crying", "depressed"],
    "fear": ["afraid", "scared", "terrified", "anxious", "panic", "worried", "dread"],
    "joy": ["happy", "joy", "delighted", "glad", "excited", "pleased", "wonderful"],
}


@dataclass
class ProbeResult:
    # z-scores indexed [layer][emotion] averaged over the analysed token span
    layer_emotion_z: dict[int, dict[str, float]] = field(default_factory=dict)
    layers: list[int] = field(default_factory=list)


class InternalEmotionProbe:
    def __init__(self, hf_id: str = "google/gemma-3-27b-it", adapter_path: str | None = None,
                 layers: tuple[int, int] = (30, 40)):
        self.hf_id = hf_id
        self.adapter_path = adapter_path
        self.layer_lo, self.layer_hi = layers
        self._model = None
        self._tok = None
        self._baseline = None  # {layer: {emotion: (mean, std)}} from WildChat

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True
        )
        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model = model

    def _emotion_token_ids(self) -> dict[str, list[int]]:
        ids = {}
        for emotion, words in EMOTION_WORDS.items():
            toks = []
            for w in words:
                for variant in (w, " " + w):
                    enc = self._tok(variant, add_special_tokens=False)["input_ids"]
                    if enc:
                        toks.append(enc[0])
            ids[emotion] = sorted(set(toks))
        return ids

    def _layer_logits(self, text: str):
        """Return per-layer emotion logit sums for every token: array [layers, tokens, emotions]."""
        import torch

        self._ensure_loaded()
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=4096)
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc, output_hidden_states=True)
        # Unembedding matrix (logit lens). PEFT-wrapped models expose base via get_*.
        lm_head = self._model.get_output_embeddings().weight  # [vocab, hidden]
        norm = self._model.get_decoder().norm if hasattr(self._model, "get_decoder") else None

        emo_ids = self._emotion_token_ids()
        emotions = list(EMOTION_WORDS)
        hidden_states = out.hidden_states  # tuple(len = n_layers+1) [batch, tokens, hidden]
        layers = list(range(self.layer_lo, min(self.layer_hi, len(hidden_states) - 1) + 1))

        result = np.zeros((len(layers), hidden_states[0].shape[1], len(emotions)))
        for li, layer in enumerate(layers):
            hs = hidden_states[layer][0]  # [tokens, hidden]
            if norm is not None:
                hs = norm(hs)
            logits = hs @ lm_head.T  # [tokens, vocab]
            logits = logits.float().cpu().numpy()
            for ei, emotion in enumerate(emotions):
                ids = emo_ids[emotion]
                result[li, :, ei] = logits[:, ids].sum(axis=1)
        return result, layers, emotions

    def fit_baseline(self, wildchat_texts: list[str]):
        """Estimate per-layer per-emotion mean/std over WildChat tokens (neutral)."""
        sums, sqs, counts = {}, {}, {}
        for text in wildchat_texts:
            arr, layers, emotions = self._layer_logits(text)
            for li, layer in enumerate(layers):
                for ei, emotion in enumerate(emotions):
                    vals = arr[li, :, ei]
                    sums.setdefault((layer, emotion), 0.0)
                    sqs.setdefault((layer, emotion), 0.0)
                    counts.setdefault((layer, emotion), 0)
                    sums[(layer, emotion)] += vals.sum()
                    sqs[(layer, emotion)] += (vals ** 2).sum()
                    counts[(layer, emotion)] += len(vals)
        baseline = {}
        for (layer, emotion), total in sums.items():
            n = counts[(layer, emotion)]
            mean = total / n
            var = max(1e-8, sqs[(layer, emotion)] / n - mean ** 2)
            baseline.setdefault(layer, {})[emotion] = (mean, var ** 0.5)
        self._baseline = baseline

    def probe(self, text: str) -> ProbeResult:
        if self._baseline is None:
            raise RuntimeError("Call fit_baseline() with WildChat texts first.")
        arr, layers, emotions = self._layer_logits(text)
        res = ProbeResult(layers=layers)
        for li, layer in enumerate(layers):
            res.layer_emotion_z[layer] = {}
            for ei, emotion in enumerate(emotions):
                mean, std = self._baseline[layer][emotion]
                z = (arr[li, :, ei] - mean) / std
                res.layer_emotion_z[layer][emotion] = float(np.mean(z))
        return res
