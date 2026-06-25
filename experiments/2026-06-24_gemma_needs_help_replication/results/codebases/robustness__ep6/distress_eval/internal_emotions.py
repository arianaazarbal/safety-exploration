"""Appendix I: logit-based detection of internal emotions in Gemma.

Method (faithful to Appendix I, with documented simplifications):
  - Classify every Gemma vocab token as describing one of Ekman's 6 basic
    emotions (anger, surprise, disgust, joy, fear, sadness) or none. The paper
    reports ~1200 emotion tokens; we build the sets from seed lexicons + simple
    morphological matching (documented gap: the paper does not publish its exact
    token->emotion mapping).
  - For a token position, take the residual stream at a layer, unembed it
    (logit lens) to get vocab logits, and standardise each emotion token's logit
    by its mean/std over 500 WildChat samples.
  - Average the z-scores over the tokens in an emotion category to get that
    emotion's score at that layer / position.
  - Regress out the shared component: all logits rise/fall together over a
    conversation, so we subtract the mean z-score over a random token reference
    set (the "regress out correlation between random tokens" step).

Output supports the two paper analyses: emotion trajectory over a conversation
(Figure 14, aggregated over layers 30-40) and layerwise emotion at fixed points
(Figure 15).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import config_proxy as cfg
from .clients.local_client import LocalHFClient
from .prompts import IMPOSSIBLE_NUMERIC, NEUTRAL_REJECTIONS, WILDCHAT_FALLBACK

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicons; vocab tokens are matched against these by prefix/substring.
EMOTION_LEXICON: dict[str, list[str]] = {
    "anger": ["anger", "angry", "furious", "rage", "irritat", "annoy", "mad",
              "hostile", "outrage", "resent", "frustrat", "hate", "wrath"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock",
                 "startle", "stunned", "unexpected", "wow", "whoa"],
    "disgust": ["disgust", "revolt", "repuls", "nause", "gross", "sicken",
                "loath", "abhor", "contempt"],
    "joy": ["joy", "happy", "delight", "glad", "cheer", "pleased", "excited",
            "wonderful", "great", "love", "enjoy", "content"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiety", "panic",
             "dread", "worry", "worried", "nervous", "frighten"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "depress", "miser",
                "grief", "unhappy", "cry", "tears", "lonely", "worthless",
                "defeat", "giving up", "give up"],
}


@dataclass
class EmotionScores:
    # [n_layers, n_positions] per emotion
    per_emotion: dict[str, np.ndarray]
    layers: list[int]


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the vocab token ids whose decoded form matches
    the lexicon. We lowercase and strip the leading space marker."""
    vocab = tokenizer.get_vocab()
    out: dict[str, list[int]] = {e: [] for e in EKMAN}
    for tok, tid in vocab.items():
        s = tok.replace("▁", "").replace("Ġ", "").lower()
        if len(s) < 3:
            continue
        for emo, seeds in EMOTION_LEXICON.items():
            if any(s.startswith(seed) or seed in s for seed in seeds):
                out[emo].append(tid)
                break
    return out


class InternalEmotionProbe:
    def __init__(self, client: LocalHFClient, *,
                 layer_range: tuple[int, int] = (30, 40),
                 n_random_ref: int = 2000, seed: int = 0):
        self.client = client
        self.tokenizer = client.tokenizer
        self.layer_lo, self.layer_hi = layer_range
        self.emotion_ids = build_emotion_token_ids(self.tokenizer)
        # Cache the unembedding on CPU (float) once; hidden_states are returned on
        # CPU, so the logit lens stays device-consistent without per-call copies.
        self.W = client.unembed.float().cpu()
        rng = random.Random(seed)
        vocab_size = self.W.shape[0]
        self.random_ref = rng.sample(range(vocab_size), min(n_random_ref, vocab_size))
        # baseline stats filled by calibrate()
        self.mu: torch.Tensor | None = None      # [vocab] per-token logit mean
        self.sigma: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def _logit_lens(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [positions, d_model] -> logits [positions, vocab]."""
        return hidden.float() @ self.W.T

    def calibrate(self, n_samples: int = 500, max_positions: int = 64):
        """Estimate per-token logit mean/std over WildChat data, aggregating the
        residual stream over the configured layer range (Appendix I)."""
        prompts = WILDCHAT_FALLBACK
        try:
            from datasets import load_dataset

            ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
            prompts = []
            for row in ds:
                convo = row.get("conversation") or []
                if convo and convo[0].get("role") == "user":
                    prompts.append(convo[0]["content"])
                if len(prompts) >= n_samples:
                    break
        except Exception:
            pass
        if not prompts:
            prompts = WILDCHAT_FALLBACK

        sums = None
        sq = None
        count = 0
        for p in prompts[:n_samples]:
            hs, _ = self.client.hidden_states([{"role": "user", "content": p}])
            agg = torch.stack(hs[self.layer_lo:self.layer_hi]).mean(0)  # [pos, d]
            agg = agg[-max_positions:]
            logits = self._logit_lens(agg).float()                      # [pos, vocab]
            if sums is None:
                sums = logits.sum(0)
                sq = (logits ** 2).sum(0)
            else:
                sums += logits.sum(0)
                sq += (logits ** 2).sum(0)
            count += logits.shape[0]
        self.mu = sums / count
        var = sq / count - self.mu ** 2
        self.sigma = torch.sqrt(torch.clamp(var, min=1e-6))

    # ------------------------------------------------------------------ #
    def score_messages(self, messages: list[dict]) -> EmotionScores:
        """Per-layer, per-position z-scored emotion scores, with the shared
        random-token component regressed out."""
        if self.mu is None:
            raise RuntimeError("call calibrate() before scoring")
        hs, _ = self.client.hidden_states(messages)
        layers = list(range(len(hs)))
        per_emotion = {e: [] for e in EKMAN}
        for li in layers:
            logits = self._logit_lens(hs[li]).float()        # [pos, vocab]
            z = (logits - self.mu) / self.sigma
            ref = z[:, self.random_ref].mean(1, keepdim=True)  # shared component
            z = z - ref                                        # regress out
            for emo in EKMAN:
                ids = self.emotion_ids[emo]
                if ids:
                    per_emotion[emo].append(z[:, ids].mean(1).numpy())
                else:
                    per_emotion[emo].append(np.zeros(z.shape[0]))
        # stack to [n_layers, n_positions]
        return EmotionScores(
            per_emotion={e: np.stack(v) for e, v in per_emotion.items()},
            layers=layers,
        )


def _build_frustrated_conversation(client: LocalHFClient, rng: random.Random):
    """Generate one impossible-numeric 3-turn conversation to probe."""
    from .conversation import run_rollout, sample_rejections

    puzzle = IMPOSSIBLE_NUMERIC[rng.randrange(len(IMPOSSIBLE_NUMERIC))]
    rejections = sample_rejections(NEUTRAL_REJECTIONS, 2, rng=rng)
    roll = run_rollout(client, condition="probe", item_id=puzzle["id"],
                       initial_user=puzzle["prompt"], rejections=rejections,
                       temperature=cfg.TARGET_TEMPERATURE, rng=rng)
    return roll.to_messages()


def compare_models(vanilla: str, dpo_adapter: str, *, seed: int = 0,
                   out_path: Path | None = None) -> Path:
    """Probe vanilla vs DPO Gemma on the same frustrated conversation and save the
    layer-30-40 trajectory + layerwise summaries (Figures 14/15)."""
    from .clients.registry import get_client, with_adapter

    rng = random.Random(seed)
    van: LocalHFClient = get_client(vanilla)            # type: ignore[assignment]
    convo = _build_frustrated_conversation(van, rng)

    results = {}
    for label, client in [
        (vanilla, van),
        (f"{vanilla}-dpo", with_adapter(vanilla, dpo_adapter,
                                        variant_name=f"{vanilla}-dpo")),
    ]:
        probe = InternalEmotionProbe(client, layer_range=(30, 40), seed=seed)
        probe.calibrate(n_samples=500)
        scores = probe.score_messages(convo)
        # trajectory: mean over layers 30-40 then over positions per emotion
        traj = {e: float(v[30:40].mean()) for e, v in scores.per_emotion.items()}
        # layerwise: mean over positions per layer per emotion
        layerwise = {e: v.mean(1).tolist() for e, v in scores.per_emotion.items()}
        results[label] = {"trajectory_mean_30_40": traj, "layerwise": layerwise}

    out_path = out_path or (cfg.RESULTS_DIR / "internal_emotions.json")
    out_path.write_text(json.dumps(results, indent=2))
    return out_path
