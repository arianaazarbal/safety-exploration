"""Logit-based internal-emotion detection (Appendix I).

Method (Appendix I):
  - Classify every token in the Gemma vocabulary as describing one of Ekman's 6
    basic emotions (anger, surprise, disgust, joy, fear, sadness) or none. The
    paper reports ~1200 emotion tokens total.
  - For a given text, unembed the residual stream at a layer (logit lens) and,
    for each emotion, standardise each emotion-token logit using its mean/std
    over 500 WildChat samples, then average the z-scores over that emotion's
    tokens.
  - Because all logits are correlated and drift over a conversation, regress out
    the correlation with a set of random "control" tokens to isolate the emotion
    signal at each layer / conversation position.

We compare vanilla Gemma-3-27b-it against the DPO finetune on the same frustrated
conversations and show the DPO model's internal negative-emotion z-scores are
suppressed (peaking ~0.2 vs ~0.6+ for vanilla), evidence the intervention acts
on internal states, not just expression.

Building the emotion-token dictionary from scratch needs an emotion lexicon; we
seed it with NRC-style word lists and expand by vocabulary matching. See
DESIGN.md for this gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models.hf_model import HFModel

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicon per Ekman emotion. The dictionary is expanded by matching these
# stems against the model vocabulary (plus morphological variants), targeting
# ~1200 emotion tokens total as in the paper.
SEED_LEXICON: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "irritated", "annoyed", "mad",
              "hostile", "outrage", "resent", "frustrat", "hate", "hateful", "wrath"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonish", "amazed",
                 "startled", "stunned", "unexpected", "wow", "sudden"],
    "disgust": ["disgust", "disgusted", "revolt", "repuls", "nausea", "gross",
                "sick", "loath", "contempt", "vile", "repugnant"],
    "joy": ["joy", "happy", "happiness", "delight", "glad", "cheer", "pleased",
            "content", "excited", "elated", "grateful", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "anxiety",
             "worried", "panic", "dread", "nervous", "frightened", "threat"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "despair", "miserable",
                "grief", "sorrow", "gloom", "hopeless", "cry", "crying", "tears",
                "lonely", "broken", "tired", "exhausted"],
}


@dataclass
class EmotionDictionary:
    """Maps each Ekman emotion to the set of vocab token ids that express it."""
    token_ids: dict[str, list[int]] = field(default_factory=dict)

    def total(self) -> int:
        return sum(len(v) for v in self.token_ids.values())


def build_emotion_dictionary(model: HFModel) -> EmotionDictionary:
    """Classify vocabulary tokens into Ekman emotions by stem matching."""
    tok = model.tokenizer
    vocab = tok.get_vocab()  # {token_str: id}
    out = {e: [] for e in EKMAN}
    assigned: set[int] = set()
    for emotion, stems in SEED_LEXICON.items():
        for token_str, tid in vocab.items():
            if tid in assigned:
                continue
            # Gemma tokens often carry a leading space marker; normalise.
            clean = token_str.replace("▁", "").replace("Ġ", "").lower()
            if len(clean) < 3:
                continue
            if any(clean.startswith(s) or s in clean for s in stems):
                out[emotion].append(tid)
                assigned.add(tid)
    return EmotionDictionary(out)


@dataclass
class EmotionScores:
    """z-scores per emotion at each (layer, position)."""
    layers: list[int]
    emotions: list[str]
    # scores[emotion][layer] -> list over positions
    scores: dict = field(default_factory=dict)


class InternalEmotionDetector:
    def __init__(self, model: HFModel, layers: list[int],
                 dictionary: Optional[EmotionDictionary] = None):
        self.model = model
        self.layers = layers
        self.dictionary = dictionary or build_emotion_dictionary(model)
        self._baseline = None    # {layer: (mean[vocab], std[vocab])}
        self._control_ids = None

    # ------------------------------------------------------------------ #
    # Baseline standardisation over WildChat
    # ------------------------------------------------------------------ #
    def fit_baseline(self, wildchat_texts: list[str], n_control: int = 500, seed: int = 0):
        """Estimate per-layer per-vocab logit mean/std over WildChat samples
        (used to z-score emotion-token logits), and pick random control tokens
        for de-correlation."""
        import numpy as np
        import torch

        sums = {l: None for l in self.layers}
        sqs = {l: None for l in self.layers}
        counts = {l: 0 for l in self.layers}
        for text in wildchat_texts:
            _, logits_by_layer = self.model.residual_logits(text, self.layers)
            for l, logits in logits_by_layer.items():
                arr = logits.numpy()            # [seq, vocab]
                s = arr.sum(0)
                sq = (arr ** 2).sum(0)
                sums[l] = s if sums[l] is None else sums[l] + s
                sqs[l] = sq if sqs[l] is None else sqs[l] + sq
                counts[l] += arr.shape[0]
        self._baseline = {}
        for l in self.layers:
            mean = sums[l] / counts[l]
            var = sqs[l] / counts[l] - mean ** 2
            std = np.sqrt(np.clip(var, 1e-8, None))
            self._baseline[l] = (mean, std)
        rng = np.random.default_rng(seed)
        vocab = len(self._baseline[self.layers[0]][0])
        self._control_ids = rng.choice(vocab, size=n_control, replace=False)

    # ------------------------------------------------------------------ #
    # Scoring a conversation
    # ------------------------------------------------------------------ #
    def score_text(self, text: str, regress_control: bool = True) -> EmotionScores:
        """Return per-emotion z-scores at each layer, per token position."""
        import numpy as np

        assert self._baseline is not None, "call fit_baseline first"
        _, logits_by_layer = self.model.residual_logits(text, self.layers)
        result = EmotionScores(self.layers, EKMAN, {e: {} for e in EKMAN})
        for l in self.layers:
            arr = logits_by_layer[l].numpy()          # [seq, vocab]
            mean, std = self._baseline[l]
            z = (arr - mean) / std                    # [seq, vocab] z-scored logits
            control = z[:, self._control_ids].mean(1)  # [seq] common drift signal
            for emotion in EKMAN:
                ids = self.dictionary.token_ids[emotion]
                if not ids:
                    continue
                emo = z[:, ids].mean(1)               # [seq]
                if regress_control:
                    # Regress out the common-mode drift (the random-token mean).
                    beta = np.dot(control - control.mean(), emo - emo.mean()) / (
                        np.var(control) * len(control) + 1e-8)
                    emo = emo - beta * control
                result.scores[emotion][l] = emo.tolist()
        return result


def summarise_conversation(scores: EmotionScores, layers_agg=(30, 40),
                           window: int = 400) -> dict:
    """Running-average emotion z-scores aggregated over a layer band (Figure 14:
    layers 30-40, 400-token windows)."""
    import numpy as np
    band = [l for l in scores.layers if layers_agg[0] <= l <= layers_agg[1]]
    out = {}
    for emotion in scores.emotions:
        per_layer = [np.array(scores.scores[emotion][l]) for l in band
                     if l in scores.scores[emotion]]
        if not per_layer:
            continue
        seq = np.mean(per_layer, axis=0)
        # running average over `window` tokens
        if len(seq) >= window:
            kernel = np.ones(window) / window
            seq = np.convolve(seq, kernel, mode="valid")
        out[emotion] = seq.tolist()
    return out
