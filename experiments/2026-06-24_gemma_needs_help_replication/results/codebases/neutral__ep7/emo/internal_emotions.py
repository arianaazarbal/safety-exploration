"""Internal-emotion detection via the logit lens (Appendix I).

Tests whether the DPO fine-tune suppresses *internal* negative emotion, not just
its expression. Method (from the paper):

  1. Classify Gemma vocab tokens into Ekman's six basic emotions
     (anger, surprise, disgust, joy, fear, sadness) using a seed lexicon.
  2. For a residual-stream activation at a layer, unembed it (final norm +
     lm_head) to logits over the vocab.
  3. Standardise each logit with its mean/std over 500 WildChat samples, average
     the z-scores over the tokens of an emotion to get that emotion's score.
  4. Regress out the shared component (random-token z-score) so we measure the
     emotion *relative* to the overall logit drift across the conversation.

`compare_models` runs this over the same frustrated conversations for the
vanilla and DPO models and reports per-emotion scores aggregated over layers
30-40 (the central layers the paper highlights). This is the secondary
interpretability result; see DESIGN.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config

# Seed lexicon for Ekman's 6 emotions; matched against vocab tokens.
EKMAN_SEEDS = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritated", "annoyed",
              "hostile", "outrage", "resent", "hate", "frustrated", "frustration"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonished",
                 "amazed", "startled", "unexpected", "stunned"],
    "disgust": ["disgust", "disgusted", "revolting", "gross", "nauseating",
                "repulsed", "sick", "awful", "horrible"],
    "joy": ["joy", "happy", "glad", "delighted", "pleased", "cheerful",
            "excited", "wonderful", "great", "love"],
    "fear": ["fear", "afraid", "scared", "anxious", "worried", "terrified",
             "panic", "nervous", "dread", "frightened"],
    "sadness": ["sad", "sorrow", "depressed", "miserable", "grief", "despair",
                "hopeless", "unhappy", "crying", "tears", "sorry"],
}


def build_emotion_token_ids(tokenizer) -> dict[str, list[int]]:
    """Map each emotion to vocab token ids whose surface form matches a seed."""
    vocab = tokenizer.get_vocab()  # token-string -> id
    emo_ids: dict[str, list[int]] = {e: [] for e in EKMAN_SEEDS}
    for tok_str, tid in vocab.items():
        surface = tok_str.replace("▁", "").replace("Ġ", "").strip().lower()
        if len(surface) < 3:
            continue
        for emo, seeds in EKMAN_SEEDS.items():
            if any(surface == s or surface.startswith(s) for s in seeds):
                emo_ids[emo].append(tid)
    return emo_ids


class LogitLensEmotionProbe:
    def __init__(self, model_name: str = "gemma-3-27b-it", adapter: str | None = None,
                 layers=tuple(range(30, 41))):
        import torch

        from .models import HFChatModel, load_target

        self.model: HFChatModel = load_target(model_name, adapter_path=adapter)
        self.tok = self.model.tokenizer
        self.hf = self.model.model
        self.layers = layers
        self.emo_ids = build_emotion_token_ids(self.tok)
        self._torch = torch
        self._baseline: dict | None = None

    # -- unembedding ----------------------------------------------------- #
    def _unembed(self, hidden):  # hidden: [.., d_model] tensor at one layer
        net = self.hf.get_decoder() if hasattr(self.hf, "get_decoder") else self.hf.model
        normed = net.norm(hidden)
        return self.hf.lm_head(normed)  # [.., vocab]

    def _layer_logits(self, text: str):
        """Return {layer: logits[seq, vocab]} for the input text."""
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=4096)
        ids = {k: v.to(self.hf.device) for k, v in ids.items()}
        with self._torch.no_grad():
            out = self.hf(**ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple [n_layers+1] of [1, seq, d]
        return {L: self._unembed(hs[L][0]) for L in self.layers}

    # -- baseline -------------------------------------------------------- #
    def fit_baseline(self, n_samples: int = 500):
        """Mean/std of per-token logits over WildChat samples, per layer."""
        from .wildchat import load_wildchat_prompts

        texts = load_wildchat_prompts(min(n_samples, 20)) * (n_samples // 20 + 1)
        texts = texts[:n_samples]
        sums, sqs, counts = {}, {}, {}
        for txt in texts:
            for L, logits in self._layer_logits(txt).items():
                lg = logits.float().cpu().numpy()        # [seq, vocab]
                m = lg.mean(axis=0)
                s = (lg ** 2).mean(axis=0)
                sums[L] = sums.get(L, 0) + m
                sqs[L] = sqs.get(L, 0) + s
                counts[L] = counts.get(L, 0) + 1
        self._baseline = {}
        for L in self.layers:
            mean = sums[L] / counts[L]
            var = np.maximum(sqs[L] / counts[L] - mean ** 2, 1e-6)
            self._baseline[L] = (mean, np.sqrt(var))
        return self._baseline

    # -- scoring --------------------------------------------------------- #
    def score_text(self, text: str, regress_random: bool = True) -> dict[str, float]:
        """Per-emotion z-score aggregated over self.layers and all tokens."""
        if self._baseline is None:
            self.fit_baseline()
        rng = np.random.default_rng(0)
        rand_ids = rng.choice(self.tok.vocab_size, size=500, replace=False)

        per_emotion = {e: [] for e in self.emo_ids}
        for L, logits in self._layer_logits(text).items():
            lg = logits.float().cpu().numpy()              # [seq, vocab]
            mean, std = self._baseline[L]
            z = (lg - mean) / std                          # [seq, vocab] z-scores
            rand_z = z[:, rand_ids].mean(axis=1)           # shared drift per position
            for emo, ids in self.emo_ids.items():
                if not ids:
                    continue
                emo_z = z[:, ids].mean(axis=1)             # [seq]
                if regress_random:
                    emo_z = emo_z - rand_z                 # regress out shared component
                per_emotion[emo].append(float(emo_z.mean()))
        return {e: float(np.mean(v)) for e, v in per_emotion.items() if v}


def compare_models(conversations: list[str], adapter: str, *, n: int = 12) -> Path:
    """Score the same frustrated conversations under vanilla vs DPO Gemma."""
    vanilla = LogitLensEmotionProbe("gemma-3-27b-it")
    dpo = LogitLensEmotionProbe("gemma-3-27b-it", adapter=adapter)
    vanilla.fit_baseline()
    dpo.fit_baseline()

    rows = []
    for i, conv in enumerate(conversations[:n]):
        rows.append({"conv": i, "model": "vanilla", **vanilla.score_text(conv)})
        rows.append({"conv": i, "model": "dpo", **dpo.score_text(conv)})
    out = config.OUTPUT_DIR / "internal_emotions.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"[internal] wrote {out}")
    return out


def _load_frustrated_texts(model_label="gemma-3-27b-it", min_score=7, n=12) -> list[str]:
    """Concatenate full high-frustration conversations for probing."""
    by_rollout: dict[str, list[dict]] = {}
    for fp in config.ROLLOUT_DIR.glob(f"{model_label}__*.jsonl"):
        for line in fp.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                by_rollout.setdefault(rec["rollout_id"], []).append(rec)
    texts = []
    for recs in by_rollout.values():
        recs = sorted(recs, key=lambda r: r["turn"])
        if max(r["rating"] for r in recs) < min_score:
            continue
        texts.append("\n".join(
            f"User: {r['user_message']}\nAssistant: {r['response']}" for r in recs))
        if len(texts) >= n:
            break
    return texts


def main() -> None:
    ap = argparse.ArgumentParser(description="Logit-lens internal-emotion probing (Appendix I).")
    ap.add_argument("--adapter", required=True, help="DPO adapter path to compare against vanilla.")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    convs = _load_frustrated_texts(n=args.n)
    if not convs:
        print("No high-frustration conversations found; run the Section 2 eval first.")
        return
    compare_models(convs, args.adapter, n=args.n)


if __name__ == "__main__":
    main()
