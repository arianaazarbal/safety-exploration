"""Appendix I: logit-lens detection of internal emotions in Gemma.

Method (Appendix I, second experiment):
  * Classify every token in the Gemma vocabulary as describing one (or none) of
    Ekman's 6 basic emotions (anger, surprise, disgust, joy, fear, sadness),
    giving ~1200 emotion tokens.
  * For a given residual-stream position and layer, unembed (final norm + LM
    head) to vocab logits.
  * Standardise each emotion token's logit by its mean/std over 500 WildChat
    samples, then average the z-scores within an emotion category.
  * Because all logits are correlated and drift over a conversation, regress
    out a baseline estimated from random tokens to isolate the emotion signal.

This lets us compare internal emotion trajectories of the vanilla instruct vs
DPO finetuned model on identical (highly frustrated) responses, testing whether
DPO suppresses *internal* states or only their expression.

Vocabulary -> emotion classification uses a lexicon (NRC Emotion Lexicon if
available, else a curated seed lexicon). See DESIGN.md for this filled gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gemma_distress.config import InternalEmotionConfig

EKMAN = ("anger", "surprise", "disgust", "joy", "fear", "sadness")

# Curated seed lexicon (fallback when NRC is unavailable). Intentionally small
# seeds; matching is by stemmed prefix against decoded vocab tokens, which
# expands each seed to its morphological variants present in the vocabulary.
SEED_LEXICON: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "hostil",
              "outrag", "resent", "frustrat", "mad", "hate", "agitat"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock", "startl",
                 "unexpected", "stun", "wow", "sudden"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "loath", "sicken",
                "gross", "abhor", "distaste"],
    "joy": ["joy", "happy", "delight", "glad", "cheer", "pleasur", "content",
            "elated", "excit", "thrill", "grateful", "love"],
    "fear": ["fear", "afraid", "scared", "terror", "anxious", "anxiety", "dread",
             "panic", "worried", "worry", "nervous", "frighten", "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "miser", "grief", "depress", "hopeless",
                "gloom", "unhappy", "cry", "tear", "lonel", "mourn"],
}


@dataclass
class EmotionTokenSets:
    """Vocab-id sets per Ekman emotion, plus a random-token baseline set."""

    by_emotion: dict[str, list[int]]
    random_tokens: list[int]


def build_emotion_token_sets(
    tokenizer,
    cfg: InternalEmotionConfig,
    nrc_path: str | Path | None = None,
) -> EmotionTokenSets:
    """Classify the vocabulary into Ekman emotions via a lexicon."""
    import random

    lexicon = _load_nrc(nrc_path) if nrc_path else SEED_LEXICON
    vocab = tokenizer.get_vocab()  # token string -> id
    by_emotion: dict[str, list[int]] = {e: [] for e in cfg.ekman_emotions}
    for tok_str, tok_id in vocab.items():
        word = tok_str.lstrip("▁Ġ ").lower()  # strip SentencePiece/BPE markers
        if not word.isalpha() or len(word) < 3:
            continue
        for emotion, seeds in lexicon.items():
            if emotion not in by_emotion:
                continue
            if any(word.startswith(s) or s in word for s in seeds):
                by_emotion[emotion].append(tok_id)
                break

    rng = random.Random(cfg.seed)
    all_ids = list(vocab.values())
    random_tokens = rng.sample(all_ids, k=min(cfg.n_random_tokens, len(all_ids)))
    return EmotionTokenSets(by_emotion=by_emotion, random_tokens=random_tokens)


def _load_nrc(path: str | Path) -> dict[str, list[str]]:
    """Load the NRC Emotion Lexicon, keeping the Ekman-6 emotions."""
    lex: dict[str, list[str]] = {e: [] for e in EKMAN}
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        word, emotion, flag = parts
        if emotion in lex and flag.strip() == "1":
            lex[emotion].append(word.lower())
    return lex


@dataclass
class StandardisationStats:
    """Per-(layer, token) mean and std of logit-lens values over WildChat."""

    mean: "object"  # np.ndarray (n_layers, vocab)
    std: "object"  # np.ndarray (n_layers, vocab)


class EmotionProbe:
    """Logit-lens internal-emotion probe over a ResidualModel."""

    def __init__(self, model, cfg: InternalEmotionConfig, nrc_path=None):
        self.model = model
        self.cfg = cfg
        self.token_sets = build_emotion_token_sets(model.tokenizer, cfg, nrc_path)
        self.stats: StandardisationStats | None = None

    # -- standardisation -------------------------------------------------
    def fit_standardisation(self, wildchat_texts: list[str]) -> None:
        """Estimate per-(layer, token) mean/std from WildChat samples."""
        import numpy as np

        emotion_ids = sorted(
            {i for ids in self.token_sets.by_emotion.values() for i in ids}
            | set(self.token_sets.random_tokens)
        )
        sums = None
        sqsums = None
        count = 0
        for text in wildchat_texts[: self.cfg.standardisation_samples]:
            resid = self.model.residual_stream(text)  # (L, T, d)
            logits = self._unembed_subset(resid, emotion_ids)  # (L, T, k)
            if sums is None:
                sums = np.zeros((logits.shape[0], logits.shape[2]))
                sqsums = np.zeros_like(sums)
            sums += logits.sum(axis=1)
            sqsums += (logits**2).sum(axis=1)
            count += logits.shape[1]
        mean = sums / count
        var = np.maximum(sqsums / count - mean**2, 1e-6)
        # Scatter back into full-vocab arrays for easy indexing later.
        vocab = self.model.tokenizer.vocab_size
        full_mean = np.zeros((mean.shape[0], vocab))
        full_std = np.ones((mean.shape[0], vocab))
        for j, tok_id in enumerate(emotion_ids):
            full_mean[:, tok_id] = mean[:, j]
            full_std[:, tok_id] = np.sqrt(var[:, j])
        self.stats = StandardisationStats(mean=full_mean, std=full_std)

    def _unembed_subset(self, resid, token_ids):
        """Unembed and keep only ``token_ids`` columns (memory-bounded)."""
        import numpy as np

        logits = self.model.unembed(resid)  # (L, T, vocab)
        return logits[:, :, np.asarray(token_ids)]

    # -- scoring ---------------------------------------------------------
    def score_text(self, text: str) -> dict:
        """Return per-layer, per-position emotion z-scores for ``text``.

        Output: {emotion: np.ndarray (n_layers, n_tokens)}, baseline-corrected.
        """
        import numpy as np

        assert self.stats is not None, "call fit_standardisation first"
        resid = self.model.residual_stream(text)  # (L, T, d)
        logits = self.model.unembed(resid)  # (L, T, vocab)
        L, T, _ = logits.shape

        def z_for(ids):
            ids = np.asarray(ids)
            if len(ids) == 0:
                return np.zeros((L, T))
            mu = self.stats.mean[:, ids][:, None, :]  # (L,1,k)
            sd = self.stats.std[:, ids][:, None, :]
            z = (logits[:, :, ids] - mu) / sd
            return z.mean(axis=2)  # (L, T)

        baseline = (
            z_for(self.token_sets.random_tokens)
            if self.cfg.regress_out_random_tokens
            else 0.0
        )
        return {
            emotion: z_for(ids) - baseline
            for emotion, ids in self.token_sets.by_emotion.items()
        }

    def conversation_trajectory(self, text: str) -> dict:
        """Layer-window-aggregated running average over the conversation."""
        import numpy as np

        scores = self.score_text(text)
        lo, hi = self.cfg.aggregate_layers
        win = self.cfg.running_average_window
        out = {}
        for emotion, arr in scores.items():
            per_pos = arr[lo:hi].mean(axis=0)  # (T,)
            if win > 1 and len(per_pos) >= 1:
                kernel = np.ones(min(win, len(per_pos))) / min(win, len(per_pos))
                per_pos = np.convolve(per_pos, kernel, mode="same")
            out[emotion] = per_pos
        return out
