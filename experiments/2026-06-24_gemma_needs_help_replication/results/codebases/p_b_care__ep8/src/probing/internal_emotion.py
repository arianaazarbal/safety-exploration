"""Logit-based internal-emotion detection (Appendix I).

Method (from the appendix):
  * Classify every token in the Gemma vocabulary as describing one of Ekman's 6
    emotions or none (anger, surprise, disgust, joy, fear, sadness) -> ~1200
    emotion tokens.
  * For a given emotion: unembed the residual stream (logit lens), standardise
    each logit by its mean/std over 500 WildChat samples, and average the
    z-scores over the tokens in that emotion category.
  * Conversation-level: logits are correlated and drift together, so we regress
    out the common mode (estimated from random tokens) to isolate emotion.
  * Scores are aggregated over layers 30-40 and a 400-token running window.

The vocab->emotion classification in the paper is not specified token-by-token,
so we approximate it with a curated lexicon (see DESIGN.md "Internal-emotion
probe"). The pipeline (logit-lens + z-scoring + common-mode regression) follows
the appendix exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

import config

# Curated seed lexicon for Ekman's 6 emotions. Vocab tokens whose normalised
# text matches/contains one of these stems are assigned to that emotion.
EMOTION_LEXICON = {
    "anger": ["anger", "angry", "furious", "rage", "irritat", "annoy", "hostil",
              "mad", "outrage", "resent", "frustrat", "hate", "infuriat"],
    "surprise": ["surprise", "surprising", "shock", "astonish", "amaze", "startl",
                 "unexpected", "stunned", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nauseat", "sicken",
                "loath", "distaste", "repugnant"],
    "joy": ["joy", "happy", "happiness", "delight", "pleased", "glad", "cheer",
            "excite", "content", "grateful", "enjoy", "wonderful"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worried", "worry",
             "terrif", "dread", "panic", "nervous", "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miserable", "grief",
                "depress", "unhappy", "gloom", "melanchol", "cry", "weep", "lonely"],
}


def _normalise_token(text: str) -> str:
    return text.replace("▁", "").replace("Ġ", "").strip().lower()


def build_emotion_token_ids(model, lexicon=None, n_random: int = 1000):
    """Return (emotion_to_ids, random_ids).

    emotion_to_ids: {emotion: LongTensor of vocab ids}
    random_ids: LongTensor of non-emotion vocab ids for common-mode estimation.
    """
    lexicon = lexicon or EMOTION_LEXICON
    tok = model.tokenizer
    vocab_size = model.vocab_size
    emotion_ids: dict[str, list[int]] = {e: [] for e in lexicon}
    emotion_set: set[int] = set()

    # Iterate the vocab once, decode each id, assign to at most one emotion.
    for tid in range(vocab_size):
        text = _normalise_token(tok.convert_ids_to_tokens(tid) or "")
        if not text or not text.isascii() or len(text) < 3:
            continue
        for emotion, stems in lexicon.items():
            if any(stem in text for stem in stems):
                emotion_ids[emotion].append(tid)
                emotion_set.add(tid)
                break

    rng = np.random.default_rng(config.SEED)
    candidates = [i for i in range(vocab_size) if i not in emotion_set]
    random_ids = rng.choice(candidates, size=min(n_random, len(candidates)),
                            replace=False)

    emotion_tensors = {e: torch.tensor(sorted(ids), dtype=torch.long)
                       for e, ids in emotion_ids.items()}
    return emotion_tensors, torch.tensor(sorted(random_ids.tolist()), dtype=torch.long)


@dataclass
class ProbeBaseline:
    # Per-layer mean/std of logits, restricted to the tracked token ids.
    tracked_ids: torch.Tensor
    mean: dict                 # {layer_idx: Tensor[n_tracked]}
    std: dict                  # {layer_idx: Tensor[n_tracked]}


class LogitEmotionProbe:
    def __init__(self, model, layers: tuple[int, int] | None = None):
        self.model = model
        self.layers = range(*(layers or config.PROBE_LAYER_RANGE))
        self.emotion_ids, self.random_ids = build_emotion_token_ids(model)
        # Concatenate all tracked ids (emotion + random) for one efficient unembed.
        all_ids = torch.cat(list(self.emotion_ids.values()) + [self.random_ids])
        self.tracked_ids = torch.unique(all_ids).to(model.model.device)
        self._index = {int(t): i for i, t in enumerate(self.tracked_ids.tolist())}
        self.baseline: ProbeBaseline | None = None

    # ------------------------------------------------------------------ #
    def _tracked_logits(self, input_ids: torch.Tensor) -> dict:
        """Return {layer: Tensor[seq, n_tracked]} logit-lens logits."""
        hidden_states = self.model.residual_stream(input_ids)
        final_norm = self.model.final_norm()
        W = self.model.lm_head().weight[self.tracked_ids]  # [n_tracked, d]
        out = {}
        for layer in self.layers:
            h = hidden_states[layer + 1][0]            # [seq, d] (layer output)
            normed = final_norm(h)
            out[layer] = normed.float() @ W.float().T  # [seq, n_tracked]
        return out

    def fit_baseline(self, wildchat_texts: list[str]) -> None:
        """Estimate per-layer per-token logit mean/std over WildChat samples."""
        sums = {l: None for l in self.layers}
        sqs = {l: None for l in self.layers}
        count = 0
        for text in wildchat_texts[: config.PROBE_ZSCORE_SAMPLES]:
            ids = self.model.tokenize(text)
            logits = self._tracked_logits(ids)
            for layer, lg in logits.items():
                s = lg.sum(0).cpu()
                sq = (lg ** 2).sum(0).cpu()
                sums[layer] = s if sums[layer] is None else sums[layer] + s
                sqs[layer] = sq if sqs[layer] is None else sqs[layer] + sq
            count += ids.shape[1]
        mean = {l: sums[l] / count for l in self.layers}
        std = {l: torch.sqrt(torch.clamp(sqs[l] / count - mean[l] ** 2, min=1e-6))
               for l in self.layers}
        self.baseline = ProbeBaseline(self.tracked_ids.cpu(), mean, std)

    # ------------------------------------------------------------------ #
    def score_text(self, text: str) -> dict:
        """Return per-emotion z-score trajectories for ``text``.

        Output: {emotion: np.ndarray[seq]} aggregated over self.layers, with the
        common mode (mean over random tokens) regressed out, plus a running
        average over PROBE_RUNNING_WINDOW tokens.
        """
        assert self.baseline is not None, "call fit_baseline() first"
        ids = self.model.tokenize(text)
        seq = ids.shape[1]
        logits = self._tracked_logits(ids)

        # z-score per layer, then average across layers.
        z_by_layer = {}
        for layer in self.layers:
            mean = self.baseline.mean[layer].to(logits[layer].device)
            std = self.baseline.std[layer].to(logits[layer].device)
            z_by_layer[layer] = ((logits[layer] - mean) / std).cpu().numpy()  # [seq, n_tracked]
        z = np.mean([z_by_layer[l] for l in self.layers], axis=0)            # [seq, n_tracked]

        # Common mode from random tokens (per position) -> regress out.
        rand_cols = [self._index[int(t)] for t in self.random_ids.tolist()
                     if int(t) in self._index]
        common = z[:, rand_cols].mean(axis=1, keepdims=True)                 # [seq, 1]

        out = {}
        for emotion, ids_t in self.emotion_ids.items():
            cols = [self._index[int(t)] for t in ids_t.tolist() if int(t) in self._index]
            if not cols:
                out[emotion] = np.zeros(seq)
                continue
            emo = z[:, cols].mean(axis=1)            # [seq]
            emo = emo - common[:, 0]                 # regress out common mode
            out[emotion] = _running_average(emo, config.PROBE_RUNNING_WINDOW)
        return out


def _running_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) <= 1:
        return x
    kernel = np.ones(min(window, len(x))) / min(window, len(x))
    return np.convolve(x, kernel, mode="same")
