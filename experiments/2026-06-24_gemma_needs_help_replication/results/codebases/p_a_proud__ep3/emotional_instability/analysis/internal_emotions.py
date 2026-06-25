"""Logit-based internal-emotion detection in Gemma (Appendix I).

Method (Appendix I):
1. Classify tokens in the Gemma vocabulary into one of Ekman's six basic
   emotions (anger, surprise, disgust, joy, fear, sadness) using an emotion
   lexicon — about 1200 emotion tokens total.
2. For a given residual stream (per layer, per position), apply the model's
   unembedding (logit lens: final norm + output embedding) to get a logit per
   token. Standardise each token's logit by its mean/std over ~500 WildChat
   samples.
3. The emotion score at a layer/position is the mean z-score over that emotion's
   tokens. To remove the global drift where *all* logits rise and fall together
   over a conversation, regress out the mean z-score of a set of random tokens
   and take the residual.

This is logit-lens probing rather than a trained linear probe (Appendix I notes
this avoids generating probe data). Architectural assumptions (final-norm name,
tied output embedding) match Gemma-3 and are guarded with clear errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..config import REPO_ROOT, InternalEmotionConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"^[▁\s]*([a-zA-Z']+)$")  # strip SentencePiece ▁ marker

# Small built-in seed lexicon (used if the NRC Emotion Lexicon is unavailable).
# Replace with data/nrc_emotion_lexicon.txt for the full ~1200-token coverage.
_SEED_LEXICON: dict[str, list[str]] = {
    "anger": [
        "angry", "anger", "rage", "furious", "mad", "irritated", "annoyed",
        "frustrated", "frustration", "hate", "hostile", "outrage", "resent",
    ],
    "surprise": [
        "surprise", "surprised", "shock", "shocked", "amazed", "astonished",
        "startled", "unexpected", "sudden", "wow",
    ],
    "disgust": [
        "disgust", "disgusted", "revolting", "gross", "nauseous", "repulsed",
        "sick", "awful", "horrible", "vile",
    ],
    "joy": [
        "joy", "happy", "glad", "delight", "pleased", "cheerful", "great",
        "wonderful", "excited", "love", "enjoy", "content",
    ],
    "fear": [
        "fear", "afraid", "scared", "terrified", "anxious", "worried", "panic",
        "dread", "nervous", "frightened", "alarmed",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "miserable", "sorry",
        "despair", "hopeless", "grief", "cry", "crying", "tired", "exhausted",
    ],
}

_NRC_PATH = REPO_ROOT / "data" / "nrc_emotion_lexicon.txt"


def build_lexicon() -> dict[str, set[str]]:
    """Word → emotion lexicon. Prefer the NRC file; fall back to the seed set."""
    if _NRC_PATH.exists():
        return _load_nrc(_NRC_PATH)
    logger.warning(
        "NRC Emotion Lexicon not found at %s; using small built-in seed lexicon. "
        "Drop the NRC lexicon there for full ~1200-token coverage.",
        _NRC_PATH,
    )
    return {emo: set(words) for emo, words in _SEED_LEXICON.items()}


def _load_nrc(path: Path) -> dict[str, set[str]]:
    # NRC format: word<TAB>emotion<TAB>0|1. We keep only Ekman's six.
    keep = set(_SEED_LEXICON)
    lex: dict[str, set[str]] = {e: set() for e in keep}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, flag = parts
            if emotion in keep and flag == "1":
                lex[emotion].add(word.lower())
    return lex


@dataclass
class EmotionTokenMap:
    token_ids: dict[str, list[int]]  # emotion -> vocab token ids
    random_ids: list[int]

    def all_emotion_ids(self) -> list[int]:
        out: list[int] = []
        for ids in self.token_ids.values():
            out.extend(ids)
        return out


def build_token_map(
    tokenizer, cfg: InternalEmotionConfig, lexicon: dict[str, set[str]] | None = None
) -> EmotionTokenMap:
    """Assign each single-word vocabulary token to (at most) one Ekman emotion."""
    import random

    lexicon = lexicon or build_lexicon()
    word_to_emotion: dict[str, str] = {}
    for emotion in cfg.ekman_emotions:
        for word in lexicon.get(emotion, set()):
            word_to_emotion.setdefault(word, emotion)

    token_ids: dict[str, list[int]] = {e: [] for e in cfg.ekman_emotions}
    vocab = tokenizer.get_vocab()
    for tok, tid in vocab.items():
        m = _WORD_RE.match(tok)
        if not m:
            continue
        word = m.group(1).lower()
        emotion = word_to_emotion.get(word)
        if emotion is not None:
            token_ids[emotion].append(tid)

    emotion_id_set = {tid for ids in token_ids.values() for tid in ids}
    rng = random.Random(0)
    all_ids = list(range(len(vocab)))
    random_ids = rng.sample(
        [i for i in all_ids if i not in emotion_id_set],
        min(cfg.n_random_tokens, len(all_ids) - len(emotion_id_set)),
    )
    counts = {e: len(ids) for e, ids in token_ids.items()}
    logger.info("Emotion token counts: %s (random=%d)", counts, len(random_ids))
    return EmotionTokenMap(token_ids=token_ids, random_ids=random_ids)


class EkmanProbe:
    """Logit-lens emotion probe over a local Gemma model."""

    def __init__(self, hf_client, cfg: InternalEmotionConfig):
        self.client = hf_client
        self.model = hf_client.model
        self.tokenizer = hf_client.tokenizer
        self.cfg = cfg
        self.token_map = build_token_map(self.tokenizer, cfg)
        self._norm, self._unembed = self._resolve_unembed()
        # Standardisation stats: per (layer, token_id) running mean/std.
        self._stats_mean: dict[int, "object"] = {}
        self._stats_std: dict[int, "object"] = {}

    def _resolve_unembed(self):
        """Return (final_norm, output_embedding_weight) for the logit lens."""
        inner = getattr(self.model, "model", self.model)
        norm = getattr(inner, "norm", None)
        out_emb = self.model.get_output_embeddings()
        if norm is None or out_emb is None:
            raise RuntimeError(
                "Could not resolve final norm / output embedding for logit lens; "
                "this probe assumes a Gemma-style architecture."
            )
        return norm, out_emb.weight

    # ------------------------------------------------------------------ #
    def _layer_logits(self, text: str, token_ids: Sequence[int]):
        """Return logit-lens values [n_layers, n_positions, n_tracked_tokens]."""
        import torch

        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple: embeddings + each layer
        idx = torch.tensor(list(token_ids), device=self.model.device)
        weight = self._unembed.index_select(0, idx)  # [n_tracked, hidden]
        layer_logits = []
        for layer in self.cfg.aggregate_layers:
            h = hidden_states[layer]  # [1, pos, hidden]
            h = self._norm(h)
            logits = torch.matmul(h[0], weight.T)  # [pos, n_tracked]
            layer_logits.append(logits.float().cpu())
        return torch.stack(layer_logits)  # [n_layers, pos, n_tracked]

    def fit_standardisation(self, wildchat_texts: Sequence[str]) -> None:
        """Compute per-(layer, token) mean/std over WildChat (Welford online)."""
        import numpy as np

        tracked = self.token_map.all_emotion_ids() + self.token_map.random_ids
        n_layers = len(self.cfg.aggregate_layers)
        n_tok = len(tracked)
        count = 0
        mean = np.zeros((n_layers, n_tok))
        m2 = np.zeros((n_layers, n_tok))

        for text in wildchat_texts[: self.cfg.standardisation_samples]:
            vals = self._layer_logits(text, tracked).numpy()  # [L, pos, n_tok]
            # Treat each position as a sample.
            for pos in range(vals.shape[1]):
                count += 1
                x = vals[:, pos, :]
                delta = x - mean
                mean += delta / count
                m2 += delta * (x - mean)
        std = np.sqrt(m2 / max(count - 1, 1))
        std[std == 0] = 1.0
        self._fit_mean = mean
        self._fit_std = std
        self._tracked = tracked
        logger.info("Fitted standardisation over %d token positions", count)

    def score_text(self, text: str):
        """Return z-scored, drift-corrected emotion scores.

        Output shape: dict ``emotion -> array[n_layers, n_positions]``.
        """
        import numpy as np

        if not hasattr(self, "_fit_mean"):
            raise RuntimeError("Call fit_standardisation() before score_text().")
        vals = self._layer_logits(text, self._tracked).numpy()  # [L, pos, n_tok]
        z = (vals - self._fit_mean[:, None, :]) / self._fit_std[:, None, :]

        # Column ranges for each emotion / random within the tracked array.
        ranges = {}
        cursor = 0
        for emotion, ids in self.token_map.token_ids.items():
            ranges[emotion] = (cursor, cursor + len(ids))
            cursor += len(ids)
        random_slice = (cursor, cursor + len(self.token_map.random_ids))

        random_mean = z[:, :, random_slice[0] : random_slice[1]].mean(axis=2)  # [L, pos]

        scores: dict[str, "np.ndarray"] = {}
        for emotion, (lo, hi) in ranges.items():
            if hi <= lo:
                scores[emotion] = np.zeros(z.shape[:2])
                continue
            emo_mean = z[:, :, lo:hi].mean(axis=2)  # [L, pos]
            if self.cfg.regress_out_random_tokens:
                emo_mean = _regress_out(emo_mean, random_mean)
            scores[emotion] = emo_mean
        return scores

    def conversation_running_average(self, text: str) -> dict[str, list[float]]:
        """Emotion scores aggregated over layers, smoothed over a token window
        (Figure 14: running average over 400-token windows)."""
        import numpy as np

        per_layer = self.score_text(text)  # emotion -> [L, pos]
        window = self.cfg.running_window_tokens
        out: dict[str, list[float]] = {}
        for emotion, arr in per_layer.items():
            layer_avg = arr.mean(axis=0)  # [pos]
            kernel = np.ones(min(window, len(layer_avg))) / min(window, max(len(layer_avg), 1))
            smoothed = np.convolve(layer_avg, kernel, mode="valid")
            out[emotion] = smoothed.tolist()
        return out


def _regress_out(signal, nuisance):
    """Return residual of ``signal`` after regressing out ``nuisance`` per layer."""
    import numpy as np

    residual = np.empty_like(signal)
    for layer in range(signal.shape[0]):
        x = nuisance[layer]
        y = signal[layer]
        if np.std(x) < 1e-8:
            residual[layer] = y - y.mean()
            continue
        beta = np.cov(x, y)[0, 1] / np.var(x)
        residual[layer] = y - beta * (x - x.mean()) - y.mean()
    return residual
