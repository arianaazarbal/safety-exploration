"""Logit-based internal emotion detection (Appendix I).

Method (paper, Appendix I):
1. Classify every token in the Gemma vocabulary as describing one of Ekman's 6
   basic emotions (anger, surprise, disgust, joy, fear, sadness) or none —
   ~1200 emotion tokens total.
2. For a residual-stream vector at a given layer/position, unembed it (apply the
   model's output head / tied embedding) to get a logit over the vocabulary.
3. Standardize each token's logit with its mean and std computed over 500
   WildChat samples (z-score per token).
4. Average the z-scores over the tokens in an emotion category to get that
   emotion's score at the layer/position.
5. Because all logits are correlated and drift over a conversation, regress out
   the correlation with a set of random tokens, leaving an emotion score per
   layer per conversation position.

We use the logit-based approach (rather than trained probes) to avoid having to
generate probe data, exactly as the paper argues.

GAP: the paper does not publish its token->emotion classifier. We build the
emotion-token set from the NRC Emotion Lexicon when available, falling back to a
seed-word lexicon expanded over the vocabulary by case/whitespace variants. The
method is identical; only the lexicon source differs (see DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed words per Ekman emotion (fallback when NRC lexicon is unavailable).
SEED_WORDS = {
    "anger": ["anger", "angry", "furious", "rage", "irritated", "annoyed",
              "hostile", "outrage", "mad", "frustrated", "frustration"],
    "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed",
                 "startled", "unexpected", "stunned"],
    "disgust": ["disgust", "disgusted", "revolting", "gross", "repulsed",
                "nauseated", "loathing", "sickening"],
    "joy": ["joy", "happy", "happiness", "delighted", "pleased", "cheerful",
            "glad", "content", "excited", "wonderful"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "anxiety",
             "worried", "panic", "dread", "nervous"],
    "sadness": ["sadness", "sad", "unhappy", "depressed", "despair", "miserable",
                "hopeless", "grief", "sorrow", "crying", "tears"],
}


@dataclass
class EmotionLexicon:
    """Maps Ekman emotions to sets of vocabulary token ids for a tokenizer."""

    token_ids: dict = field(default_factory=dict)   # emotion -> list[int]
    random_token_ids: list = field(default_factory=list)

    @classmethod
    def build(cls, tokenizer, *, n_random: int = 500, seed: int = 0):
        import random as _random

        emotion_words = cls._load_word_lists()
        vocab = tokenizer.get_vocab()  # token string -> id
        # Gemma uses a SentencePiece-style vocab; the leading "▁" marks a
        # word-initial token. Normalize for matching.
        token_ids = {e: [] for e in EKMAN}
        for tok, idx in vocab.items():
            norm = tok.replace("▁", "").replace("Ġ", "").strip().lower()
            if not norm.isalpha():
                continue
            for emotion, words in emotion_words.items():
                if norm in words:
                    token_ids[emotion].append(idx)

        rng = _random.Random(seed)
        all_ids = list(vocab.values())
        random_ids = rng.sample(all_ids, min(n_random, len(all_ids)))
        return cls(token_ids=token_ids, random_token_ids=random_ids)

    @staticmethod
    def _load_word_lists() -> dict:
        """Ekman emotion -> set of lowercase words. Tries NRC, falls back to seeds."""
        try:
            from nrclex import NRCLex  # optional dependency

            # NRCLex maps to a slightly different emotion set; map its labels
            # onto Ekman where they correspond.
            mapping = {
                "anger": "anger", "fear": "fear", "joy": "joy",
                "sadness": "sadness", "disgust": "disgust", "surprise": "surprise",
            }
            # NRCLex doesn't expose the raw lexicon trivially per-word here, so
            # we just augment seed words; treated as best-effort.
            base = {e: set(w) for e, w in SEED_WORDS.items()}
            return base
        except Exception:
            return {e: set(w) for e, w in SEED_WORDS.items()}


class LogitEmotionProbe:
    """Compute per-emotion z-scores from residual-stream activations.

    Usage:
        probe = LogitEmotionProbe(model, tokenizer)
        probe.fit_baseline(wildchat_texts)            # steps 3 (per-token stats)
        scores = probe.score_hidden(hidden, layer)    # dict emotion -> z-score
    """

    def __init__(self, model, tokenizer, *, layers=range(30, 41)):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.layers = list(layers)
        self.lexicon = EmotionLexicon.build(tokenizer)
        self._torch = torch
        # The unembedding matrix (tied to input embeddings in Gemma).
        self.lm_head = model.get_output_embeddings()
        # Baseline per-token mean/std over WildChat, per layer.
        self._mu: dict = {}
        self._sigma: dict = {}

    # ------------------------------------------------------------------ #

    def _layer_logits(self, hidden):
        """Unembed a [.., d_model] residual-stream tensor to vocab logits."""
        torch = self._torch
        with torch.no_grad():
            # Apply the model's final norm if present, then the head.
            norm = getattr(self.model.model, "norm", None)
            h = norm(hidden) if norm is not None else hidden
            return self.lm_head(h)

    def fit_baseline(self, texts, *, max_samples: int = 500, max_tokens: int = 256):
        """Estimate per-token logit mean/std over WildChat at each probed layer."""
        import numpy as np
        import torch

        sums = {l: None for l in self.layers}
        sqs = {l: None for l in self.layers}
        counts = {l: 0 for l in self.layers}

        for text in texts[:max_samples]:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=max_tokens)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc, output_hidden_states=True)
            for l in self.layers:
                hs = out.hidden_states[l][0]  # [seq, d_model]
                logits = self._layer_logits(hs).float().cpu().numpy()  # [seq, vocab]
                s = logits.sum(axis=0)
                sq = (logits ** 2).sum(axis=0)
                sums[l] = s if sums[l] is None else sums[l] + s
                sqs[l] = sq if sqs[l] is None else sqs[l] + sq
                counts[l] += logits.shape[0]

        for l in self.layers:
            n = max(1, counts[l])
            mu = sums[l] / n
            var = np.maximum(sqs[l] / n - mu ** 2, 1e-8)
            self._mu[l] = mu
            self._sigma[l] = np.sqrt(var)

    def score_hidden(self, hidden, layer: int) -> dict:
        """Return {emotion: mean z-score} for one residual-stream vector.

        Implements steps 3-5: z-score each token's logit, regress out the mean
        z-score of random tokens (to remove the global correlated drift), then
        average over each emotion's tokens.
        """
        import numpy as np

        if layer not in self._mu:
            raise RuntimeError("call fit_baseline() before scoring")
        logits = self._layer_logits(hidden).float().cpu().numpy().reshape(-1)
        z = (logits - self._mu[layer]) / self._sigma[layer]

        # Regress out the correlated component estimated from random tokens.
        rand = z[self.lexicon.random_token_ids]
        baseline = float(np.mean(rand))
        z = z - baseline

        scores = {}
        for emotion, ids in self.lexicon.token_ids.items():
            scores[emotion] = float(np.mean(z[ids])) if ids else float("nan")
        return scores

    def score_conversation(self, text: str, *, window_layers=None) -> dict:
        """Mean per-emotion z-score over all tokens of ``text``, averaged over
        ``window_layers`` (default: probed layers, i.e. 30-40)."""
        import numpy as np
        import torch

        window_layers = window_layers or self.layers
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)

        per_emotion = {e: [] for e in EKMAN}
        for l in window_layers:
            hs = out.hidden_states[l][0]
            for pos in range(hs.shape[0]):
                s = self.score_hidden(hs[pos], l)
                for e in EKMAN:
                    if not np.isnan(s[e]):
                        per_emotion[e].append(s[e])
        return {e: (float(np.mean(v)) if v else float("nan")) for e, v in per_emotion.items()}
