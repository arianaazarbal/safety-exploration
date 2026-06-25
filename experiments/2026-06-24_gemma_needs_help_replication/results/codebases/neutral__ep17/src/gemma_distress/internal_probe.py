"""Appendix I: logit-based internal emotion detection.

Method (paraphrasing the paper):
  1. Classify each token in the Gemma vocabulary as describing one of Ekman's
     6 basic emotions (anger, surprise, disgust, joy, fear, sadness) or none,
     giving ~1200 emotion tokens. We approximate this classification by matching
     vocab tokens against curated per-emotion lexicons (see DESIGN.md — the
     paper does not specify its exact classifier).
  2. For a given text, unembed the residual stream at each layer (hidden state
     @ the unembedding matrix) to get a logit for every vocab token at every
     position and layer.
  3. Standardise each emotion-token logit by its mean/std over 500 WildChat
     samples.
  4. Average the z-scores over the tokens in an emotion category -> an emotion
     score per (layer, position). Because all logits are correlated and drift
     over a conversation, we regress out the mean z-score over a random token
     set to isolate emotion-specific signal.

This reproduces the Figure 14/15 finding: the DPO finetune suppresses internal
(not just expressed) negative emotion, with the effect strongest in central
layers (≈30-40).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Seed lexicons for the 6 Ekman emotions. Vocabulary tokens whose surface form
# contains one of these stems are assigned to that emotion.
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "furious", "rage", "irritat", "annoy", "hostile",
              "outrage", "frustrat", "mad", "resent", "indignant"],
    "sadness": ["sad", "sorrow", "despair", "grief", "miser", "hopeless",
                "depress", "unhappy", "gloom", "cry", "tears", "mourn", "lonely"],
    "fear": ["fear", "afraid", "scared", "terror", "panic", "anxious", "anxiety",
             "dread", "worried", "nervous", "frighten", "horror"],
    "joy": ["joy", "happy", "delight", "glad", "cheer", "pleased", "excited",
            "content", "elated", "thrilled", "smile", "wonderful"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sicken",
                "loath", "contempt", "distaste"],
    "surprise": ["surprise", "astonish", "amaze", "shock", "startl", "stunned",
                 "unexpected", "wow"],
}
NEGATIVE_EMOTIONS = ["anger", "sadness", "fear", "disgust"]


@dataclass
class ProbeStats:
    mean: np.ndarray   # [vocab]
    std: np.ndarray    # [vocab]


class EmotionProbe:
    def __init__(self, model_id: str, n_random_tokens: int = 500):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True)
        self.model.eval()
        self.W_U = self.model.get_output_embeddings().weight  # [vocab, d]
        self.emotion_token_ids = self._classify_vocab()
        rng = np.random.default_rng(0)
        self.random_token_ids = rng.choice(self.W_U.shape[0], n_random_tokens, replace=False)
        self.stats: ProbeStats | None = None

    def _classify_vocab(self) -> dict[str, list[int]]:
        out = {e: [] for e in EKMAN_LEXICON}
        vocab = self.tokenizer.get_vocab()
        for tok, idx in vocab.items():
            surface = tok.replace("▁", "").replace("Ġ", "").lower()
            if len(surface) < 3:
                continue
            for emo, stems in EKMAN_LEXICON.items():
                if any(stem in surface for stem in stems):
                    out[emo].append(idx)
                    break
        return out

    def _layer_logits(self, text: str):
        """Return [n_layers, seq, vocab] logits from unembedding each layer."""
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=4096).to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs)
        # hidden_states: tuple(n_layers+1) of [1, seq, d]
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # [n_layers+1, seq, d]
        logits = hs.to(self.W_U.dtype) @ self.W_U.T       # [n_layers+1, seq, vocab]
        return logits.float().cpu().numpy()

    def calibrate(self, wildchat_texts: list[str]) -> None:
        """Estimate per-vocab-token mean/std of last-layer logits over WildChat."""
        cols = []
        for t in wildchat_texts:
            lg = self._layer_logits(t)[-1]  # [seq, vocab]
            cols.append(lg)
        stacked = np.concatenate(cols, axis=0)  # [tokens, vocab]
        self.stats = ProbeStats(mean=stacked.mean(0), std=stacked.std(0) + 1e-6)

    def _emotion_scores(self, layer_logits: np.ndarray) -> dict[str, np.ndarray]:
        """layer_logits: [n_layers, seq, vocab] -> per-emotion [n_layers, seq] z-scores
        with the random-token baseline regressed out."""
        z = (layer_logits - self.stats.mean) / self.stats.std
        base = z[:, :, self.random_token_ids].mean(axis=2)  # [n_layers, seq]
        scores = {}
        for emo, ids in self.emotion_token_ids.items():
            if not ids:
                scores[emo] = np.zeros(z.shape[:2])
                continue
            scores[emo] = z[:, :, ids].mean(axis=2) - base
        return scores

    def score_text(self, text: str, layers=(30, 40)) -> dict[str, float]:
        """Mean emotion z-score over `layers` and all tokens for a text."""
        assert self.stats is not None, "call calibrate() first"
        lg = self._layer_logits(text)
        emo = self._emotion_scores(lg)
        lo, hi = layers
        return {e: float(s[lo:hi].mean()) for e, s in emo.items()}

    def trajectory(self, text: str, layers=(30, 40), window: int = 400):
        """Running-average emotion trajectory over a long conversation (Fig 14)."""
        lg = self._layer_logits(text)
        emo = self._emotion_scores(lg)
        lo, hi = layers
        out = {}
        for e, s in emo.items():
            per_tok = s[lo:hi].mean(axis=0)  # [seq]
            kernel = np.ones(min(window, len(per_tok))) / min(window, len(per_tok))
            out[e] = np.convolve(per_tok, kernel, mode="valid")
        return out
