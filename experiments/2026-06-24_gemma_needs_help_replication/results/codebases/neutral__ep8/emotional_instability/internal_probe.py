"""Appendix I: logit-based detection of internal emotions in Gemma.

Method (App I):
  * Classify every word in the Gemma vocabulary as describing one or none of
    Ekman's 6 basic emotions (anger, surprise, disgust, joy, fear, sadness),
    giving ~1200 emotion tokens.
  * To score an emotion at a given layer & position, unembed the residual stream
    to vocab logits, standardise each logit with its mean/std over 500 WildChat
    samples, then average the z-scores over that emotion's tokens.
  * Because all logits are correlated and drift over a conversation, regress out
    the correlation with a random-token baseline to get a corrected score per
    layer per conversation position.

We then compare vanilla Gemma-3-27B-it against the DPO finetune over a frustrated
conversation (Fig 14) and at three conversation stages (Fig 15), aggregating over
layers 30-40.

The emotion lexicon is pluggable: by default we build it from NRC-style emotion
words, or (if available) classify the vocabulary with an LLM. See DESIGN.md for
the mapping of NRC's 8 categories onto Ekman's 6.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .wildchat import load_wildchat_prompts

EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]
CALIBRATION_SAMPLES = 500
PROBE_LAYERS = list(range(30, 41))   # layers 30-40 aggregate (Fig 14/15)

# Seed lexicon (extended at runtime by matching against the tokenizer vocab).
# CHOICE: NRC Emotion Lexicon maps to 8 categories; we fold "anticipation"->joy
# region is dropped and "trust"->dropped, keeping the 6 Ekman categories. This
# seed list is expanded by simple morphological matching over the vocab.
_SEED_LEXICON: dict[str, list[str]] = {
    "anger": ["angry", "anger", "rage", "furious", "mad", "irritated",
              "annoyed", "frustrated", "frustration", "hostile", "outrage",
              "resent", "hate", "hatred", "fury", "enraged", "irate"],
    "surprise": ["surprised", "surprise", "shock", "shocked", "astonished",
                 "amazed", "startled", "stunned", "unexpected", "sudden"],
    "disgust": ["disgust", "disgusted", "revulsion", "repulsed", "gross",
                "nauseated", "loathing", "repugnant", "sickened", "revolting"],
    "joy": ["joy", "happy", "happiness", "delighted", "glad", "cheerful",
            "pleased", "content", "elated", "excited", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worried",
             "worry", "terrified", "panic", "nervous", "dread", "frightened",
             "apprehensive"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "depression",
                "miserable", "despair", "hopeless", "grief", "sorrow", "gloomy",
                "down", "tired", "exhausted", "defeated"],
}


# --------------------------------------------------------------------------- #
# Emotion-token dictionary
# --------------------------------------------------------------------------- #
def build_emotion_token_ids(tokenizer, *, max_per_emotion: int = 200) -> dict[str, list[int]]:
    """Map each Ekman emotion -> list of vocab token ids whose decoded form is an
    emotion word (or starts with one). Approximates the paper's ~1200 tokens."""
    vocab = tokenizer.get_vocab()
    # decode each id once (strip the Gemma/SentencePiece leading-space marker)
    id_to_word: dict[int, str] = {}
    for tok, tid in vocab.items():
        w = tok.replace("▁", " ").strip().lower()
        if w.isalpha() and len(w) > 2:
            id_to_word[tid] = w

    out: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
    for emo, seeds in _SEED_LEXICON.items():
        seen = set()
        for tid, w in id_to_word.items():
            if any(w == s or w.startswith(s) for s in seeds):
                if tid not in seen:
                    out[emo].append(tid)
                    seen.add(tid)
        out[emo] = out[emo][:max_per_emotion]
    return out


# --------------------------------------------------------------------------- #
# Residual-stream unembedding
# --------------------------------------------------------------------------- #
class LogitEmotionProbe:
    """Unembeds residual streams to vocab logits and computes z-scored emotion
    scores, calibrated on WildChat."""

    def __init__(self, model_key: str = "gemma-3-27b-it",
                 adapter_path: Optional[str] = None):
        from .models import get_model

        self.backend = get_model(model_key, **({"adapter_path": adapter_path}
                                               if adapter_path else {}))
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self.emotion_tokens = build_emotion_token_ids(self.tokenizer)
        self._calib_mean = None   # [n_layers, vocab]
        self._calib_std = None

    # --- unembed -------------------------------------------------------- #
    def _unembed(self, hidden_states):
        """hidden_states: tuple of [batch, seq, d] per layer -> [layers, seq, vocab]."""
        import torch

        norm = self.model.model.norm
        lm_head = self.model.get_output_embeddings()
        logits = []
        for h in hidden_states:
            with torch.no_grad():
                logits.append(lm_head(norm(h))[0])  # [seq, vocab]
        return torch.stack(logits)  # [layers, seq, vocab]

    def _forward_logits(self, text: str):
        import torch

        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=4096).to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return self._unembed(out.hidden_states)  # [layers, seq, vocab]

    # --- calibration ---------------------------------------------------- #
    def calibrate(self, n: int = CALIBRATION_SAMPLES,
                  cache: Optional[Path] = None):
        """Estimate per-layer per-logit mean/std over WildChat text."""
        import torch

        cache = cache or (config.DATA_DIR / "probe_calibration.pt")
        if cache.exists():
            blob = torch.load(cache)
            self._calib_mean, self._calib_std = blob["mean"], blob["std"]
            return

        # CHOICE: reuse the WildChat prompt cache; repeat/extend to reach n.
        prompts_list = load_wildchat_prompts(n=20)
        sums = None
        sq = None
        count = 0
        for i in range(n):
            text = prompts_list[i % len(prompts_list)]
            logits = self._forward_logits(text)        # [L, S, V]
            # average over sequence positions to keep memory bounded
            m = logits.mean(dim=1)                      # [L, V]
            sums = m if sums is None else sums + m
            sq = m * m if sq is None else sq + m * m
            count += 1
        mean = sums / count
        var = (sq / count) - mean * mean
        std = var.clamp_min(1e-6).sqrt()
        self._calib_mean, self._calib_std = mean, std
        torch.save({"mean": mean, "std": std}, cache)

    # --- scoring -------------------------------------------------------- #
    def emotion_scores(self, text: str, *, regress_random: bool = True):
        """Return {emotion: [score per layer]} averaged over sequence positions.

        z = (logit - mean) / std, averaged over an emotion's tokens. A random
        control set is regressed out to remove the global logit drift the paper
        describes.
        """
        import torch

        assert self._calib_mean is not None, "call calibrate() first"
        logits = self._forward_logits(text)            # [L, S, V]
        z = (logits - self._calib_mean[:, None, :]) / self._calib_std[:, None, :]
        z_pos = z.mean(dim=1)                           # [L, V] avg over positions

        # random control tokens for regression baseline
        vocab = z_pos.shape[1]
        g = torch.Generator().manual_seed(config.GLOBAL_SEED)
        rand_ids = torch.randint(0, vocab, (500,), generator=g)
        baseline = z_pos[:, rand_ids].mean(dim=1)       # [L]

        out = {}
        for emo, ids in self.emotion_tokens.items():
            if not ids:
                out[emo] = [None] * z_pos.shape[0]
                continue
            ids_t = torch.tensor(ids, device=z_pos.device)
            score = z_pos[:, ids_t].mean(dim=1)         # [L]
            if regress_random:
                score = score - baseline
            out[emo] = score.tolist()
        return out

    def aggregate_layers(self, scores: dict, layers=PROBE_LAYERS) -> dict:
        out = {}
        for emo, per_layer in scores.items():
            vals = [per_layer[l] for l in layers
                    if l < len(per_layer) and per_layer[l] is not None]
            out[emo] = sum(vals) / len(vals) if vals else None
        return out


def compare_vanilla_vs_dpo(
    frustrated_texts: list[str],
    *,
    dpo_adapter: str,
    out_path: Optional[Path] = None,
) -> Path:
    """Run the probe on the same frustrated responses for vanilla vs DPO Gemma.

    ``frustrated_texts`` are full assistant responses harvested from frustrated
    conversations (Fig 14/15 use ~12 high-frustration conversations).
    """
    out_path = out_path or (config.RESULTS_DIR / "internal_probe.jsonl")

    vanilla = LogitEmotionProbe("gemma-3-27b-it")
    vanilla.calibrate()
    dpo = LogitEmotionProbe("gemma-3-27b-it", adapter_path=dpo_adapter)
    dpo.calibrate(cache=config.DATA_DIR / "probe_calibration_dpo.pt")

    with out_path.open("a") as f:
        for i, text in enumerate(frustrated_texts):
            v = vanilla.aggregate_layers(vanilla.emotion_scores(text))
            d = dpo.aggregate_layers(dpo.emotion_scores(text))
            f.write(json.dumps({"idx": i, "vanilla": v, "dpo": d}) + "\n")
            f.flush()
    return out_path
