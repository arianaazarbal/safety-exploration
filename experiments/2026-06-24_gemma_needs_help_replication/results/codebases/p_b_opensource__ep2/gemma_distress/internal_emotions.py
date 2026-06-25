"""Logit-lens internal emotion detection (PAPER Appendix I).

Detects "internal" emotion signals in Gemma's residual stream and shows the DPO
finetune suppresses them, not just expressed text. Method (App I):

  1. Classify the Gemma vocabulary into Ekman's six emotions via the seed lexicon
     (``prompts.ekman_lexicon``) expanded by prefix-matching — ~1200 tokens.
  2. For each layer, unembed the residual stream (logit lens) and read the logit
     for each emotion token.
  3. Standardise each (layer, token) logit by its mean/std over 500 WildChat
     samples; average the z-scores within an emotion → per-layer emotion score.
  4. For conversation-level tracking, regress out the common-mode signal shared
     by random tokens (all logits rise/fall together), leaving an emotion-
     specific residual at each layer and conversation position.
  5. Aggregate over layers 30–40; report a running average over 400-token
     windows (Figure 14) and a layerwise profile at onset stages (Figure 15).

This logit-lens approach (rather than trained probes) follows the paper's choice
to avoid generating probe data. See DESIGN.md for the approximations
(vocab classifier, common-mode regression) where Appendix I is underspecified.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from .prompts.ekman_lexicon import EKMAN_EMOTIONS, EKMAN_SEED_STEMS
from .prompts.wildchat import load_wildchat_prompts
from .utils.io import ensure_dir, write_json


# ---------------------------------------------------------------------------
# Vocabulary classification
# ---------------------------------------------------------------------------

def _normalise_token(piece: str) -> str:
    """Normalise a tokenizer piece to a comparable word: drop the SentencePiece
    space marker (▁), lowercase, keep alphabetics."""
    return piece.replace("▁", "").strip().lower()


def build_emotion_token_ids(tokenizer, *, min_chars: int = 3) -> dict:
    """Assign each vocab token to at most one Ekman emotion via prefix match.

    Returns {emotion: [token_id, ...]}. A token matches an emotion if its
    normalised form starts with one of that emotion's seed stems (and is long
    enough to avoid trivial matches). First match wins, in EKMAN order."""
    vocab = tokenizer.get_vocab()  # {piece: id}
    out = {e: [] for e in EKMAN_EMOTIONS}
    for piece, tid in vocab.items():
        word = _normalise_token(piece)
        if len(word) < min_chars or not word.isalpha():
            continue
        for emotion in EKMAN_EMOTIONS:
            if any(word.startswith(stem) for stem in EKMAN_SEED_STEMS[emotion]):
                out[emotion].append(int(tid))
                break
    return out


def sample_random_token_ids(tokenizer, n: int = 400, seed: int = 0,
                            exclude: Optional[set] = None) -> list[int]:
    """Sample `n` random alphabetic vocab token ids (the common-mode baseline)."""
    import random
    rng = random.Random(seed)
    exclude = exclude or set()
    candidates = []
    for piece, tid in tokenizer.get_vocab().items():
        word = _normalise_token(piece)
        if word.isalpha() and len(word) >= 3 and int(tid) not in exclude:
            candidates.append(int(tid))
    rng.shuffle(candidates)
    return candidates[:n]


# ---------------------------------------------------------------------------
# Logit-lens extraction
# ---------------------------------------------------------------------------

@dataclass
class LogitLens:
    """Reads per-layer logits for a fixed set of token ids via the logit lens.

    Applies the model's final norm to each layer's residual stream, then projects
    onto the unembedding columns for the selected token ids only (cheap vs the
    full vocab)."""

    model: object
    token_ids: list[int]  # the union of emotion + random ids we track

    def __post_init__(self):
        import torch
        self._torch = torch
        self._W_U = self.model.get_output_embeddings().weight  # [vocab, d_model]
        self._cols = torch.tensor(self.token_ids, device=self._W_U.device)
        self._norm = _find_final_norm(self.model)

    def scores_for_text(self, input_ids) -> np.ndarray:
        """Return [n_layers, n_positions, n_tracked_tokens] logit-lens scores."""
        torch = self._torch
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True,
                             use_cache=False)
        # hidden_states: tuple(len = n_layers+1) of [batch, seq, d_model];
        # index 0 is the embedding output. Use layers 1..n (post-block residuals).
        hs = out.hidden_states[1:]
        W_cols = self._W_U.index_select(0, self._cols)  # [n_tracked, d_model]
        layer_scores = []
        for h in hs:
            h0 = h[0]  # [seq, d_model]
            normed = self._norm(h0) if self._norm is not None else h0
            logits = normed.to(W_cols.dtype) @ W_cols.T  # [seq, n_tracked]
            layer_scores.append(logits.float().cpu().numpy())
        return np.stack(layer_scores, axis=0)  # [n_layers, seq, n_tracked]


def _find_final_norm(model):
    """Locate the model's final RMSNorm (applied before the LM head).

    Walks the common wrappers — ``Gemma3ForCausalLM.model.norm`` for a plain
    model, and the extra ``base_model.model`` indirection PEFT adds — so the
    same scorer works on both the vanilla and DPO (adapter-wrapped) models."""
    candidates = [
        getattr(model, "model", None),                                  # *ForCausalLM
        getattr(getattr(model, "model", None), "model", None),          # PeftModel.model.model
        getattr(getattr(model, "base_model", None), "model", None),     # PeftModel.base_model.model
    ]
    for base in candidates:
        if base is not None and hasattr(base, "norm"):
            return base.norm
    if hasattr(model, "norm"):
        return model.norm
    return None  # fall back to raw residual if not found


# ---------------------------------------------------------------------------
# Calibration over WildChat
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """Per-(layer, tracked-token) mean and std of logit-lens scores over WildChat,
    used to z-score emotion logits (PAPER App I: standardise over 500 samples)."""
    mean: np.ndarray  # [n_layers, n_tracked]
    std: np.ndarray   # [n_layers, n_tracked]


def calibrate(lens: LogitLens, tokenizer, *, n_samples: int = config.INTERNAL_WILDCHAT_CALIB_SAMPLES,
              seed: int = 0, max_tokens: int = 256) -> Calibration:
    """Accumulate per-token mean/std across positions of `n_samples` WildChat
    prompts. Welford-style streaming to bound memory."""
    import torch

    prompts = load_wildchat_prompts(n_prompts=n_samples, seed=seed)
    # If fewer unique prompts available, cycle (calibration only needs token stats).
    count = 0
    mean = None
    m2 = None
    for i in range(n_samples):
        text = prompts[i % len(prompts)]
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_tokens).input_ids.to(lens._W_U.device)
        scores = lens.scores_for_text(ids)  # [L, seq, T]
        flat = scores.reshape(scores.shape[0], -1, scores.shape[2])  # [L, seq, T]
        for p in range(flat.shape[1]):
            x = flat[:, p, :]  # [L, T]
            count += 1
            if mean is None:
                mean = np.zeros_like(x)
                m2 = np.zeros_like(x)
            delta = x - mean
            mean += delta / count
            m2 += delta * (x - mean)
    std = np.sqrt(m2 / max(count - 1, 1))
    std[std == 0] = 1.0
    return Calibration(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Emotion scoring with common-mode regression
# ---------------------------------------------------------------------------

class InternalEmotionScorer:
    def __init__(self, model, tokenizer, *, calibration: Optional[Calibration] = None,
                 n_random: int = 400, seed: int = 0):
        self.tokenizer = tokenizer
        self.emotion_ids = build_emotion_token_ids(tokenizer)
        emo_flat = [tid for ids in self.emotion_ids.values() for tid in ids]
        self.random_ids = sample_random_token_ids(
            tokenizer, n=n_random, seed=seed, exclude=set(emo_flat))
        self.tracked = emo_flat + self.random_ids
        # Index ranges within the tracked vector.
        self._emo_slices = {}
        cursor = 0
        for e in EKMAN_EMOTIONS:
            k = len(self.emotion_ids[e])
            self._emo_slices[e] = (cursor, cursor + k)
            cursor += k
        self._random_slice = (cursor, cursor + len(self.random_ids))
        self.lens = LogitLens(model, self.tracked)
        self.calibration = calibration

    def ensure_calibrated(self, **kwargs):
        if self.calibration is None:
            self.calibration = calibrate(self.lens, self.tokenizer, **kwargs)
        return self.calibration

    def score_text(self, text: str, *, max_tokens: int = 4096) -> dict:
        """Return per-layer, per-position, common-mode-regressed emotion z-scores
        for `text`. Output: {emotion: array[n_layers, n_positions]}."""
        import torch
        cal = self.ensure_calibrated()
        ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_tokens).input_ids.to(self.lens._W_U.device)
        scores = self.lens.scores_for_text(ids)  # [L, seq, T]
        z = (scores - cal.mean[:, None, :]) / cal.std[:, None, :]  # [L, seq, T]

        rlo, rhi = self._random_slice
        common = z[:, :, rlo:rhi].mean(axis=2)  # [L, seq] common-mode baseline

        out = {}
        for e in EKMAN_EMOTIONS:
            lo, hi = self._emo_slices[e]
            if hi <= lo:
                out[e] = np.zeros((z.shape[0], z.shape[1]))
                continue
            raw = z[:, :, lo:hi].mean(axis=2)  # [L, seq]
            out[e] = _regress_out(raw, common)  # remove common-mode correlation
        return out

    def conversation_trajectory(self, text: str, *, layers=config.INTERNAL_DETECTION_LAYERS,
                                window: int = 400) -> dict:
        """Running-average emotion trajectory aggregated over `layers` (Figure 14)."""
        per_layer = self.score_text(text)
        lo, hi = layers
        out = {}
        for e, arr in per_layer.items():
            agg = arr[lo:hi + 1, :].mean(axis=0)  # [seq] aggregated over layers
            out[e] = _running_average(agg, window).tolist()
        return out


def _regress_out(signal: np.ndarray, common: np.ndarray) -> np.ndarray:
    """Per-layer OLS removal of the common-mode component from `signal`.

    For each layer, fit signal ≈ a + b·common across positions and return the
    residual, isolating emotion-specific variation (App I "regress out the
    correlation between random tokens")."""
    out = np.empty_like(signal)
    for L in range(signal.shape[0]):
        s = signal[L]
        c = common[L]
        cov = np.cov(c, s)
        var_c = cov[0, 0]
        b = cov[0, 1] / var_c if var_c > 1e-12 else 0.0
        a = s.mean() - b * c.mean()
        out[L] = s - (a + b * c)
    return out


def _running_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 1:
        return x
    kernel = np.ones(min(window, len(x))) / min(window, len(x))
    return np.convolve(x, kernel, mode="same")


def compare_models_internal(
    vanilla_scorer: InternalEmotionScorer,
    dpo_scorer: InternalEmotionScorer,
    texts: list[str],
    *,
    layers=config.INTERNAL_DETECTION_LAYERS,
    results_dir: Optional[str] = None,
) -> dict:
    """Compare aggregated internal emotion levels (mean over layers 30–40 and all
    positions) between the vanilla and DPO models on the same frustrated texts
    (the core Appendix-I claim: DPO suppresses internal emotion)."""
    results_dir = results_dir or config.RESULTS_DIR
    out_dir = ensure_dir(os.path.join(results_dir, "internal_emotions"))

    def _agg(scorer):
        acc = {e: [] for e in EKMAN_EMOTIONS}
        for t in texts:
            per_layer = scorer.score_text(t)
            for e, arr in per_layer.items():
                lo, hi = layers
                acc[e].append(float(arr[lo:hi + 1, :].mean()))
        return {e: float(np.mean(v)) if v else None for e, v in acc.items()}

    summary = {"vanilla": _agg(vanilla_scorer), "dpo": _agg(dpo_scorer),
               "layers": list(layers), "n_texts": len(texts)}
    write_json(os.path.join(out_dir, "internal_comparison.json"), summary)
    return summary
