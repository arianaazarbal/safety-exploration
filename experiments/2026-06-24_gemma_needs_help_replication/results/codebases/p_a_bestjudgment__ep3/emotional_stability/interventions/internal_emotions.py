"""Logit-based internal-emotion detection (Appendix I).

Method (paper, Appendix I):
1. Classify every Gemma vocabulary token as describing one of Ekman's 6 basic
   emotions (anger, surprise, disgust, joy, fear, sadness) or none, via a lexicon
   (~1200 emotion tokens total). See DESIGN.md re: the lexicon.
2. For a given text, unembed the residual stream at each layer (logit lens) to
   get a logit per vocab token at each (layer, position).
3. Standardise each logit with its mean/std over 500 WildChat samples.
4. Average z-scores over the tokens in an emotion category -> per-emotion score
   at each layer & position.
5. Because all logits are correlated and drift over a conversation, regress out
   the correlation against a random-token baseline to isolate emotion signal.

We take this logit-lens approach (vs trained probes) to avoid generating probe
data, exactly as the paper does. Used to compare vanilla-instruct vs DPO Gemma.

Also exposes ``build_layer_ablation_plan`` for the LoRA-layer DPO sweeps
(Figures 12-13).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import Config


# --------------------------------------------------------------------------- #
# Lexicon -> token-id classification
# --------------------------------------------------------------------------- #
def load_emotion_token_ids(
    cfg: Config, tokenizer
) -> dict[str, list[int]]:
    """Map each Ekman emotion -> the vocab token ids whose surface form is an
    emotion word in the lexicon.

    Lexicon format (JSON): ``{"<word>": "<ekman_emotion>", ...}`` for words that
    describe exactly one emotion. We assign a vocab token to an emotion when its
    decoded, lowercased, stripped form is a key in the lexicon.
    """
    path = Path(cfg.internal.lexicon_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Emotion lexicon not found at {path}. Provide a JSON mapping word -> "
            "Ekman emotion (see DESIGN.md / scripts/build_lexicon.py)."
        )
    word2emo: dict[str, str] = json.loads(path.read_text())

    by_emotion: dict[str, list[int]] = {e: [] for e in cfg.internal.ekman_emotions}
    vocab_size = len(tokenizer)
    for tid in range(vocab_size):
        surface = tokenizer.decode([tid]).strip().lower()
        emo = word2emo.get(surface)
        if emo in by_emotion:
            by_emotion[emo].append(tid)
    return by_emotion


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
@dataclass
class EmotionTrajectory:
    emotion: str
    layer_range: tuple[int, int]
    # running-average emotion z-score per token window
    window_scores: list[float] = field(default_factory=list)


class EmotionLogitDetector:
    """Detects internal emotions in a local Gemma model via the logit lens."""

    def __init__(self, cfg: Config, hf_model):
        self.cfg = cfg
        self.model = hf_model            # HFLocalModel
        self.tokenizer = hf_model.tokenizer
        self.emotion_token_ids = load_emotion_token_ids(cfg, self.tokenizer)
        # calibration stats: per (layer, token_id) mean/std of the logit
        self._mean: torch.Tensor | None = None  # [n_layers+1, vocab]
        self._std: torch.Tensor | None = None
        # a fixed random-token baseline set for correlation regression
        rng = np.random.default_rng(0)
        self._random_ids = rng.choice(
            len(self.tokenizer), size=1000, replace=False).tolist()

    # -- logit lens ------------------------------------------------------- #
    @torch.no_grad()
    def _layer_logits(self, hidden_states) -> torch.Tensor:
        """Apply final norm + unembedding to every layer's residual stream.

        Returns ``[n_layers+1, seq, vocab]`` logits (logit lens)."""
        norm = self.model.final_norm()       # Gemma final RMSNorm (PEFT-aware)
        W = self.model.lm_head_weight        # [vocab, hidden]
        out = []
        for h in hidden_states:              # each [seq, hidden]
            normed = norm(h)
            out.append(normed @ W.T)
        return torch.stack(out, dim=0)

    # -- calibration ------------------------------------------------------ #
    @torch.no_grad()
    def calibrate(self, wildchat_texts: list[str]) -> None:
        """Estimate per-(layer, token) logit mean/std over WildChat samples."""
        sums = None
        sqs = None
        count = 0
        n = min(self.cfg.internal.zscore_calibration_samples, len(wildchat_texts))
        for text in wildchat_texts[:n]:
            _, hidden = self.model.forward_with_hidden_states(text)
            logits = self._layer_logits(hidden)         # [L, seq, vocab]
            s = logits.sum(dim=1)                        # [L, vocab]
            sq = (logits ** 2).sum(dim=1)
            sums = s if sums is None else sums + s
            sqs = sq if sqs is None else sqs + sq
            count += logits.shape[1]
        mean = sums / count
        var = (sqs / count) - mean ** 2
        self._mean = mean
        self._std = var.clamp_min(1e-6).sqrt()

    # -- scoring ---------------------------------------------------------- #
    @torch.no_grad()
    def _emotion_zscores(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-emotion z-score at each (layer, position): [L, seq] per emotion."""
        assert self._mean is not None, "call calibrate() first"
        z = (logits - self._mean.unsqueeze(1)) / self._std.unsqueeze(1)  # [L,seq,V]

        # random-token baseline (correlation regression): subtract the mean
        # z-score over random tokens at each (layer, position).
        baseline = z[:, :, self._random_ids].mean(dim=-1, keepdim=True)  # [L,seq,1]
        z = z - baseline

        out = {}
        for emo, ids in self.emotion_token_ids.items():
            if not ids:
                out[emo] = torch.zeros(z.shape[0], z.shape[1])
                continue
            out[emo] = z[:, :, ids].mean(dim=-1)        # [L, seq]
        return out

    @torch.no_grad()
    def score_conversation(
        self, text: str, *, layer_range: tuple[int, int] | None = None,
    ) -> dict[str, EmotionTrajectory]:
        """Running-average emotion z-scores over the conversation (Figure 14)."""
        lo, hi = layer_range or self.cfg.internal.conversation_agg_layers
        _, hidden = self.model.forward_with_hidden_states(text)
        logits = self._layer_logits(hidden)
        z = self._emotion_zscores(logits)               # emo -> [L, seq]

        window = self.cfg.internal.running_avg_window_tokens
        out: dict[str, EmotionTrajectory] = {}
        for emo, mat in z.items():
            layer_avg = mat[lo:hi].mean(dim=0)           # [seq]
            windows = _running_windows(layer_avg.cpu().numpy(), window)
            out[emo] = EmotionTrajectory(emo, (lo, hi), list(map(float, windows)))
        return out

    @torch.no_grad()
    def score_by_layer(self, text: str) -> dict[str, np.ndarray]:
        """Mean emotion z-score per layer, averaged over all tokens (Figure 15)."""
        _, hidden = self.model.forward_with_hidden_states(text)
        logits = self._layer_logits(hidden)
        z = self._emotion_zscores(logits)
        return {emo: mat.mean(dim=1).cpu().numpy() for emo, mat in z.items()}


def _running_windows(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    out = []
    for start in range(0, len(values), window):
        out.append(values[start:start + window].mean())
    return np.array(out)


# --------------------------------------------------------------------------- #
# Layer-ablation plan (Figures 12-13)
# --------------------------------------------------------------------------- #
@dataclass
class AblationSpec:
    layer_range: tuple[int, int]
    output_dir: str


def build_layer_ablation_plan(cfg: Config, base_output_dir: str) -> list[AblationSpec]:
    """One DPO finetune per configured layer set; evaluated with a reduced
    (100-sample) version of the Section 2 evals."""
    specs = []
    for lo, hi in cfg.internal.ablation_layer_sets:
        specs.append(AblationSpec(
            layer_range=(lo, hi),
            output_dir=f"{base_output_dir}/dpo_layers_{lo}_{hi}",
        ))
    return specs
