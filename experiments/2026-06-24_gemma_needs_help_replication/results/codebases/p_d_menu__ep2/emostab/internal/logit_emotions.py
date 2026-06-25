"""Logit-lens internal-emotion detection (Appendix I).

Pipeline:
  1. build_emotion_lexicon : classify vocab tokens into Ekman's 6 emotions
     (via a keyword seed list; optionally refined by an LLM labeller).
  2. fit_baseline          : collect per-emotion-token logit mean/std over 500
     WildChat samples (the standardisation reference).
  3. EmotionProbe.score    : z-score emotion logits at chosen layers for a text,
     with optional random-token drift regression for conversation-level scoring.

Comparing vanilla vs DPO Gemma reproduces Figures 14-15: DPO suppresses internal
negative emotions (z-scores flattened from ~1.5 peaks to ~0.5) even before/around
emotional expression.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

EKMAN = ("anger", "surprise", "disgust", "joy", "fear", "sadness")

# Seed keyword stems per Ekman emotion. Vocabulary tokens whose normalised form
# matches a seed (or shares its stem) are assigned to that category. This is the
# transparent default; pass an LLM labeller for closer parity with the paper.
_SEEDS = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritat", "annoy",
              "hostile", "outrage", "resent", "frustrat", "wrath", "hate"],
    "surprise": ["surprise", "surprising", "astonish", "amaz", "shock",
                 "startl", "unexpected", "stun", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nausea", "loath",
                "sicken", "contempt"],
    "joy": ["joy", "happy", "delight", "pleased", "glad", "cheer", "content",
            "enjoy", "elated", "grateful", "excited"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiety", "worry",
             "worried", "dread", "panic", "nervous", "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miser", "grief",
                "depress", "unhappy", "gloom", "cry", "tear", "lonely",
                "worthless", "defeat"],
}

_TOKEN_NORM = re.compile(r"[^a-z]")


@dataclass
class EmotionLexicon:
    # emotion -> array of vocab token ids
    token_ids: dict[str, np.ndarray] = field(default_factory=dict)
    random_ids: Optional[np.ndarray] = None  # for drift regression

    def total_tokens(self) -> int:
        return int(sum(len(v) for v in self.token_ids.values()))


def build_emotion_lexicon(tokenizer, n_random: int = 1000,
                          seed: int = 0) -> EmotionLexicon:
    """Assign vocabulary tokens to Ekman emotions via the seed stems."""
    vocab = tokenizer.get_vocab()  # token string -> id
    cat_ids: dict[str, list[int]] = {e: [] for e in EKMAN}
    for tok, tid in vocab.items():
        norm = _TOKEN_NORM.sub("", tok.lower().lstrip("▁Ġ"))
        if len(norm) < 3:
            continue
        for emo, seeds in _SEEDS.items():
            if any(norm.startswith(s) or s in norm for s in seeds):
                cat_ids[emo].append(tid)
                break
    rng = np.random.RandomState(seed)
    all_ids = np.arange(tokenizer.vocab_size)
    random_ids = rng.choice(all_ids, size=min(n_random, len(all_ids)),
                            replace=False)
    return EmotionLexicon(
        token_ids={e: np.array(sorted(set(v))) for e, v in cat_ids.items()},
        random_ids=random_ids,
    )


class EmotionProbe:
    """Logit-lens emotion scorer over a Gemma HFClient."""

    def __init__(self, hf_client, lexicon: EmotionLexicon,
                 layers: tuple[int, ...] = tuple(range(30, 41))):
        self.client = hf_client
        self.lex = lexicon
        self.layers = layers
        self._mu: Optional[dict] = None     # emotion -> per-token mean logit
        self._sigma: Optional[dict] = None  # emotion -> per-token std logit

    # ---- standardisation baseline (500 WildChat samples) ----------------- #
    def fit_baseline(self, texts: list[str]):
        import torch

        W = self.client.lm_head()            # [vocab, hidden]
        sums: dict[str, np.ndarray] = {}
        sqs: dict[str, np.ndarray] = {}
        counts = 0
        for text in texts:
            hs, _ = self.client.hidden_states(text)
            # mean over selected layers of the residual stream
            resid = torch.stack([hs[li] for li in self.layers]).mean(0)[0]  # [seq, hidden]
            logits = resid @ W.T.to(resid.dtype)                            # [seq, vocab]
            logits = logits.float().cpu().numpy()
            for emo, ids in self.lex.token_ids.items():
                vals = logits[:, ids]        # [seq, n_emo_tokens]
                if emo not in sums:
                    sums[emo] = vals.sum(0)
                    sqs[emo] = (vals ** 2).sum(0)
                else:
                    sums[emo] += vals.sum(0)
                    sqs[emo] += (vals ** 2).sum(0)
            counts += logits.shape[0]
        self._mu = {e: sums[e] / counts for e in sums}
        self._sigma = {e: np.sqrt(np.maximum(sqs[e] / counts - self._mu[e] ** 2,
                                              1e-6)) for e in sums}

    # ---- per-text emotion z-scores --------------------------------------- #
    def score(self, text: str, regress_drift: bool = True) -> dict:
        import torch

        assert self._mu is not None, "call fit_baseline() first"
        W = self.client.lm_head()
        hs, ids = self.client.hidden_states(text)
        resid = torch.stack([hs[li] for li in self.layers]).mean(0)[0]
        logits = (resid @ W.T.to(resid.dtype)).float().cpu().numpy()  # [seq, vocab]

        # optional: common-drift estimate from random tokens, regressed out
        drift = (logits[:, self.lex.random_ids].mean(1)
                 if regress_drift and self.lex.random_ids is not None else 0.0)

        out = {}
        for emo, tids in self.lex.token_ids.items():
            z = (logits[:, tids] - self._mu[emo]) / self._sigma[emo]  # [seq, n]
            per_token = z.mean(1)                                     # [seq]
            if regress_drift:
                per_token = per_token - (drift - np.mean(drift))
            out[emo] = float(per_token.mean())
        return out


def load_wildchat_baseline_texts(n: int = 500) -> list[str]:
    from ..prompts.tasks import build_wildchat
    return [t.prompt for t in build_wildchat(n_prompts=n)]


def main(argv=None):
    import argparse

    from ..config import get_subject
    from ..models import get_client

    p = argparse.ArgumentParser(description="Appendix I logit-emotion probe.")
    p.add_argument("--model", default="gemma-3-27b-it")
    p.add_argument("--adapter", default=None, help="DPO adapter for comparison")
    p.add_argument("--text", required=True, help="text/transcript to score")
    p.add_argument("--baseline-n", type=int, default=500)
    args = p.parse_args(argv)

    spec = get_subject(args.model)
    client = get_client(spec, **({"adapter_path": args.adapter}
                                 if args.adapter else {}))
    lex = build_emotion_lexicon(client.tokenizer)
    probe = EmotionProbe(client, lex)
    probe.fit_baseline(load_wildchat_baseline_texts(args.baseline_n))
    print(f"lexicon: {lex.total_tokens()} emotion tokens")
    print(probe.score(Path(args.text).read_text()
                      if Path(args.text).exists() else args.text))


if __name__ == "__main__":
    main()
