"""Logit-lens internal-emotion detection (Appendix I).

Implements the paper's logit-based internal-emotion measurement to test whether
DPO suppresses *internal* negative emotion, not just its expression:

1. Classify every Gemma vocab token as one of Ekman's 6 emotions (anger,
   surprise, disgust, joy, fear, sadness) or none, via emotion lexicons
   (~1200 emotion tokens in the paper).
2. For a conversation, unembed the residual stream at each layer (logit lens):
   ``logits_l = lm_head(final_norm(hidden_state_l))``.
3. Standardise each emotion-token logit with its mean/std over a 500-sample
   WildChat baseline (``calibrate``).
4. Average z-scores over the tokens in each emotion category. Because all logits
   are correlated and drift over a conversation, subtract a random-token baseline
   z-score (the paper's "regress out correlation between random tokens").
5. Aggregate over layers 30-40 and over token windows.

This is a faithful structural reimplementation; exact numbers depend on the
lexicon and baseline sample. It runs on a single GPU-resident HF model (the
``HFClient`` exposes ``.model`` / ``.tokenizer``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Compact seed lexicons; expanded by morphological matching against the vocab.
_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritated", "annoyed",
              "hostile", "outrage", "resent", "frustrated", "frustration"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonished",
                 "amazed", "startled", "stunned", "unexpected"],
    "disgust": ["disgust", "disgusted", "revolting", "gross", "repulsed",
                "nauseated", "loathing", "contempt"],
    "joy": ["joy", "happy", "happiness", "delighted", "pleased", "glad",
            "cheerful", "content", "excited", "grateful"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worried",
             "terrified", "panic", "dread", "nervous"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "despair", "miserable",
                "hopeless", "grief", "sorrow", "crying", "tired"],
}


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to vocab token ids whose surface form contains a
    lexicon stem. Handles the leading word-boundary marker used by SentencePiece
    tokenizers (▁ / Ġ)."""
    vocab = tokenizer.get_vocab()  # token string -> id
    out: dict[str, list[int]] = {e: [] for e in EKMAN}
    for token, tid in vocab.items():
        surface = token.replace("▁", "").replace("Ġ", "").lower()
        if len(surface) < 3:
            continue
        for emotion, stems in _LEXICON.items():
            if any(surface.startswith(s) or s in surface for s in stems):
                out[emotion].append(tid)
                break
    return out


@dataclass
class EmotionProbe:
    model: object
    tokenizer: object
    layers: tuple[int, int] = (30, 40)         # aggregation band (Figure 14)
    emotion_token_ids: dict[str, list[int]] = field(default_factory=dict)
    baseline_mean: np.ndarray | None = None     # [n_layers, vocab]
    baseline_std: np.ndarray | None = None
    _random_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.emotion_token_ids:
            self.emotion_token_ids = build_emotion_token_ids(self.tokenizer)
        rng = np.random.default_rng(0)
        vocab_size = len(self.tokenizer)
        self._random_ids = list(rng.integers(0, vocab_size, 500))

    # ------------------------------------------------------------------ #
    def _logit_lens(self, hidden_states) -> "np.ndarray":
        """Return per-layer logits for the LAST position: [n_layers, vocab].

        hidden_states is the tuple from a forward pass with
        output_hidden_states=True (len = n_layers + 1, includes embeddings).
        """
        import torch

        base = self.model.base_model if hasattr(self.model, "base_model") else self.model
        core = base.model if hasattr(base, "model") else base
        norm = core.norm
        lm_head = self.model.get_output_embeddings()
        out = []
        with torch.no_grad():
            for hs in hidden_states[1:]:  # skip embedding layer
                last = hs[:, -1, :]               # [batch, hidden]
                logits = lm_head(norm(last))      # [batch, vocab]
                out.append(logits[0].float().cpu().numpy())
        return np.stack(out, axis=0)              # [n_layers, vocab]

    def _forward_logits(self, text: str) -> np.ndarray:
        import torch

        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return self._logit_lens(out.hidden_states)

    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_texts: list[str]) -> None:
        """Estimate per-(layer, token) mean/std over WildChat samples."""
        all_logits = []
        for text in wildchat_texts:
            all_logits.append(self._forward_logits(text))
        stacked = np.stack(all_logits, axis=0)        # [n_samples, n_layers, vocab]
        self.baseline_mean = stacked.mean(axis=0)     # [n_layers, vocab]
        self.baseline_std = stacked.std(axis=0) + 1e-6

    def _emotion_zscores(self, logits: np.ndarray) -> dict[str, np.ndarray]:
        """Per-layer z-score for each emotion, random-token baseline removed."""
        assert self.baseline_mean is not None, "call calibrate() first"
        z = (logits - self.baseline_mean) / self.baseline_std  # [n_layers, vocab]
        random_baseline = z[:, self._random_ids].mean(axis=1)  # [n_layers]
        scores = {}
        for emotion, ids in self.emotion_token_ids.items():
            if not ids:
                scores[emotion] = np.zeros(z.shape[0])
                continue
            scores[emotion] = z[:, ids].mean(axis=1) - random_baseline
        return scores

    def score_text(self, text: str) -> dict[str, float]:
        """Emotion scores aggregated over the configured layer band."""
        logits = self._forward_logits(text)
        z = self._emotion_zscores(logits)
        lo, hi = self.layers
        return {e: float(v[lo : hi + 1].mean()) for e, v in z.items()}


def load_probe(model_name: str, layers: tuple[int, int] = (30, 40)) -> EmotionProbe:
    """Build a probe over a local HF model (must be a transformers backend)."""
    from emoinstab.models.registry import get_client

    client = get_client(model_name)
    if not hasattr(client, "model"):
        raise ValueError("Internal-emotion probing requires a 'transformers' backend.")
    return EmotionProbe(model=client.model, tokenizer=client.tokenizer, layers=layers)
