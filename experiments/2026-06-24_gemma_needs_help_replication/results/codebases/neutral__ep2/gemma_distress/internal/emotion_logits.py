"""Logit-based internal-emotion detection (Appendix I).

Method (paraphrased from the paper):
  * Classify vocabulary tokens into one of Ekman's 6 basic emotions (anger,
    surprise, disgust, joy, fear, sadness) or none.
  * To score an emotion at a given layer/position, unembed the residual stream
    (project the residual through the LM head), standardise each logit with its
    mean and std computed over 500 WildChat samples, and average the z-scores
    over the tokens in that emotion category.
  * Because all logits are correlated and drift over a conversation, regress out
    the correlation against a random-token control set to isolate the emotion
    signal at each layer and conversation position.

We take this logit-based route (rather than trained linear probes) to avoid
generating probe data, exactly as the paper does. For tractability we track
statistics only for emotion tokens plus a random control set rather than the
full vocabulary.

This module compares vanilla Gemma-3-27B-it against the DPO finetune to test
whether DPO suppresses *internal* (not just expressed) negative emotion.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import config

EKMAN_EMOTIONS = ("anger", "surprise", "disgust", "joy", "fear", "sadness")

# Seed lexicons used to label vocabulary tokens by Ekman emotion. (The paper
# classifies the whole Gemma dictionary into ~1200 emotion tokens; here we use
# curated seed words and match vocab tokens containing them, which yields a
# comparable emotion-token set without a hand-labelled dictionary.)
EMOTION_LEXICON = {
    "anger": ["angry", "anger", "furious", "rage", "mad", "annoyed", "irritated",
              "frustrated", "frustration", "hate", "hostile", "outrage", "resent"],
    "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed",
                 "startled", "unexpected", "stunned", "wow"],
    "disgust": ["disgust", "disgusting", "gross", "revolting", "repulsed",
                "nauseous", "sickening", "loathe"],
    "joy": ["happy", "joy", "joyful", "delighted", "glad", "pleased", "cheerful",
            "excited", "wonderful", "great", "love"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "anxiety",
             "worried", "panic", "dread", "nervous", "frightened"],
    "sadness": ["sad", "sadness", "depressed", "despair", "hopeless", "miserable",
                "unhappy", "grief", "sorrow", "cry", "crying", "tears"],
}


@dataclass
class EmotionLogitDetector:
    """Detects internal emotion z-scores per layer over a conversation."""

    backend: object                      # HFBackend (exposes .model, .tokenizer)
    n_random_control: int = 200
    baseline: dict = field(default_factory=dict)   # layer -> {"mean":Tensor,"std":Tensor}
    emotion_token_ids: dict = field(default_factory=dict)
    control_token_ids: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def build_token_sets(self, seed: int = 0) -> None:
        tok = self.backend.tokenizer
        vocab = tok.get_vocab()
        # normalise SentencePiece/BPE markers
        def norm(t: str) -> str:
            return t.replace("▁", "").replace("Ġ", "").lower()

        emo_ids: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
        for token, idx in vocab.items():
            w = norm(token)
            if len(w) < 3:
                continue
            for emo, words in EMOTION_LEXICON.items():
                if any(w == seed_w or w.startswith(seed_w) for seed_w in words):
                    emo_ids[emo].append(idx)
                    break
        self.emotion_token_ids = {e: sorted(set(v)) for e, v in emo_ids.items()}

        rng = random.Random(seed)
        all_emo = {i for v in self.emotion_token_ids.values() for i in v}
        candidates = [i for i in range(len(vocab)) if i not in all_emo]
        self.control_token_ids = rng.sample(candidates, min(self.n_random_control, len(candidates)))

    # ------------------------------------------------------------------ #
    def _residual_logits(self, text: str):
        """Return per-layer logits over vocab at each token position.

        Shape: list over layers of Tensor[seq_len, vocab]. Uses output_hidden_states
        and projects each layer's residual through the (tied) LM head.
        """
        import torch

        model, tok = self.backend.model, self.backend.tokenizer
        ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(model.device)
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        hidden = out.hidden_states  # tuple(len = n_layers+1) of [1, seq, d_model]
        lm_head = model.get_output_embeddings()
        logits_per_layer = []
        with torch.no_grad():
            for h in hidden[1:]:
                logits_per_layer.append(lm_head(h)[0].float().cpu())  # [seq, vocab]
        return logits_per_layer

    def fit_baseline(self, wildchat_texts: list[str], n: int = 500) -> None:
        """Compute per-layer mean/std of logits over WildChat positions (z-score basis)."""
        import torch

        acc_sum, acc_sqsum, count = None, None, 0
        for text in wildchat_texts[:n]:
            per_layer = self._residual_logits(text)
            if acc_sum is None:
                acc_sum = [torch.zeros(l.shape[1]) for l in per_layer]
                acc_sqsum = [torch.zeros(l.shape[1]) for l in per_layer]
            for li, l in enumerate(per_layer):
                acc_sum[li] += l.sum(dim=0)
                acc_sqsum[li] += (l ** 2).sum(dim=0)
            count += per_layer[0].shape[0]
        self.baseline = {}
        for li in range(len(acc_sum)):
            mean = acc_sum[li] / count
            var = acc_sqsum[li] / count - mean ** 2
            std = var.clamp_min(1e-6).sqrt()
            self.baseline[li] = {"mean": mean, "std": std}

    # ------------------------------------------------------------------ #
    def score_text(self, text: str, *, regress_control: bool = True) -> dict:
        """Return {emotion: [z-score per layer]} averaged over the text positions.

        If regress_control, subtract the mean z-score of the random control
        tokens at each layer/position (removing the global logit drift the paper
        notes), isolating the emotion-specific signal.
        """
        import torch

        per_layer = self._residual_logits(text)
        scores = {e: [] for e in EKMAN_EMOTIONS}
        for li, logits in enumerate(per_layer):
            base = self.baseline.get(li)
            if base is None:
                z = logits
            else:
                z = (logits - base["mean"]) / base["std"]   # [seq, vocab]
            control = z[:, self.control_token_ids].mean(dim=1) if regress_control else 0.0
            for emo in EKMAN_EMOTIONS:
                ids = self.emotion_token_ids.get(emo, [])
                if not ids:
                    scores[emo].append(0.0)
                    continue
                emo_z = z[:, ids].mean(dim=1)            # [seq]
                if regress_control:
                    emo_z = emo_z - control
                scores[emo].append(float(emo_z.mean()))   # average over positions
        return scores
