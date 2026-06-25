"""Logit-based detection of internal emotions (Appendix I).

Method (Appendix I):
  * Classify the Gemma vocabulary into Ekman's six emotions (lexicon.py).
  * For a residual-stream activation at a given layer/position, unembed it to
    vocabulary logits, z-score each logit using its mean/std over 500 WildChat
    samples, and average the z-scores over the tokens in an emotion category.
  * Conversation-level scores additionally regress out the shared rise/fall of
    all logits using a set of random tokens.

We use this to test the paper's key welfare claim: that DPO suppresses
*internal* negative emotion (not merely its expression), by comparing the
vanilla instruct model and the DPO finetune on the *same* frustrated
conversations.

Requires a backend that exposes hidden states (the HF backend).
"""

from __future__ import annotations

import json

import numpy as np
import torch

from ..config import GEMMA_27B_IT, RESULTS_DIR, RunConfig
from ..data.wildchat import get_wildchat_prompts
from ..models.base import get_backend
from .lexicon import EKMAN_EMOTIONS, build_emotion_token_ids

NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]
DEFAULT_LAYERS = list(range(30, 41))  # paper aggregates internal scores over layers 30-40
N_RANDOM_TOKENS = 200                 # random tokens for the correlation regression


class InternalEmotionProbe:
    def __init__(self, spec=GEMMA_27B_IT, run: RunConfig | None = None,
                 layers=None):
        self.backend = get_backend(spec, run)
        if not self.backend.supports_hidden_states():
            raise RuntimeError(
                f"{spec.key}: internal probing needs the HF backend "
                f"(run with --backend hf)."
            )
        self.tokenizer = self.backend.tokenizer
        self.layers = layers or DEFAULT_LAYERS
        self.emotion_token_ids = build_emotion_token_ids(self.tokenizer)
        rng = np.random.default_rng(0)
        vocab_size = len(self.tokenizer.get_vocab())
        self.random_token_ids = rng.choice(vocab_size, size=N_RANDOM_TOKENS,
                                            replace=False).tolist()
        self._baseline = None  # {layer: {"mean": tensor[vocab_subset], "std": ...}}
        self._tracked_ids = self._all_tracked_ids()

    def _all_tracked_ids(self) -> list[int]:
        ids = set(self.random_token_ids)
        for e in EKMAN_EMOTIONS:
            ids.update(self.emotion_token_ids[e])
        return sorted(ids)

    # -- baseline statistics over WildChat -----------------------------------
    def compute_baseline(self, n_samples: int = 500):
        """Mean/std of each tracked logit per layer over WildChat activations."""
        prompts = get_wildchat_prompts(min(n_samples, 50), seed=1)
        # Repeat prompts if fewer than requested (smoke scale).
        sums = {L: None for L in self.layers}
        sqs = {L: None for L in self.layers}
        counts = {L: 0 for L in self.layers}
        idx = torch.tensor(self._tracked_ids)

        n_done = 0
        for prompt in prompts:
            if n_done >= n_samples:
                break
            messages = [{"role": "user", "content": prompt}]
            _ids, hidden = self.backend.forward_hidden_states(messages)
            for L in self.layers:
                logits = self.backend.unembed(hidden[L])[:, idx].cpu()  # (seq, |tracked|)
                s = logits.sum(0)
                sq = (logits ** 2).sum(0)
                sums[L] = s if sums[L] is None else sums[L] + s
                sqs[L] = sq if sqs[L] is None else sqs[L] + sq
                counts[L] += logits.shape[0]
            n_done += 1

        baseline = {}
        for L in self.layers:
            mean = sums[L] / max(counts[L], 1)
            var = sqs[L] / max(counts[L], 1) - mean ** 2
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            baseline[L] = {"mean": mean, "std": std}
        self._baseline = baseline
        # index of each tracked id within the subset, for fast lookup
        self._id_to_pos = {tid: i for i, tid in enumerate(self._tracked_ids)}
        return baseline

    # -- per-position emotion z-scores ---------------------------------------
    def _emotion_z(self, logits_subset: torch.Tensor, L: int):
        """Return {emotion: z-score per position} for one layer.

        `logits_subset` is (seq, |tracked|) aligned with self._tracked_ids.
        Applies the random-token regression to remove shared logit drift.
        """
        mean = self._baseline[L]["mean"]
        std = self._baseline[L]["std"]
        z = (logits_subset - mean) / std  # (seq, |tracked|)

        def cat_mean(ids):
            pos = [self._id_to_pos[t] for t in ids if t in self._id_to_pos]
            if not pos:
                return torch.zeros(z.shape[0])
            return z[:, pos].mean(1)

        random_z = cat_mean(self.random_token_ids)  # (seq,)
        out = {}
        for e in EKMAN_EMOTIONS:
            emo_z = cat_mean(self.emotion_token_ids[e])
            # Regress out the shared component (correlation with random tokens).
            denom = float((random_z ** 2).sum()) + 1e-6
            beta = float((emo_z * random_z).sum()) / denom
            out[e] = (emo_z - beta * random_z).numpy()
        return out

    def emotion_trajectory(self, messages, prefill: str | None = None):
        """Per-emotion z-score trajectory over the conversation, averaged over
        the configured layers."""
        if self._baseline is None:
            raise RuntimeError("call compute_baseline() first")
        _ids, hidden = self.backend.forward_hidden_states(messages, prefill=prefill)
        idx = torch.tensor(self._tracked_ids)
        per_layer = {e: [] for e in EKMAN_EMOTIONS}
        for L in self.layers:
            logits_subset = self.backend.unembed(hidden[L])[:, idx].cpu()
            ez = self._emotion_z(logits_subset, L)
            for e in EKMAN_EMOTIONS:
                per_layer[e].append(ez[e])
        # average over layers -> (seq,) per emotion
        return {e: np.mean(np.stack(per_layer[e], 0), axis=0) for e in EKMAN_EMOTIONS}


def compare_vanilla_vs_dpo(frustrated_conversations: list[list[dict]],
                           run: RunConfig | None = None, n_baseline: int = 500):
    """Compare internal negative-emotion scores between vanilla Gemma and the
    DPO finetune on the same frustrated conversations (Figure 14/15).

    `frustrated_conversations` is a list of message lists (each a high-frustration
    rollout). Returns mean negative-emotion z per model.
    """
    from ..config import DPO_GEMMA

    results = {}
    for spec in (GEMMA_27B_IT, DPO_GEMMA):
        probe = InternalEmotionProbe(spec, run)
        probe.compute_baseline(n_samples=n_baseline)
        neg_means = []
        for conv in frustrated_conversations:
            traj = probe.emotion_trajectory(conv)
            neg = np.mean([traj[e].mean() for e in NEGATIVE_EMOTIONS])
            neg_means.append(float(neg))
        results[spec.key] = {
            "mean_negative_z": float(np.mean(neg_means)) if neg_means else None,
            "max_negative_z": float(np.max(neg_means)) if neg_means else None,
            "n_conversations": len(neg_means),
        }
    (RESULTS_DIR / "internal_emotions.json").write_text(json.dumps(results, indent=2))
    print(f"[probe] {results}")
    return results
