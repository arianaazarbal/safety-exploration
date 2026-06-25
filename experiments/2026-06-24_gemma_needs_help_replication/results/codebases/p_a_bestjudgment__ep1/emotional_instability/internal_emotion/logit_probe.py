"""Logit-based internal emotion detection (Appendix I).

Method (paraphrased from Appendix I): over the Gemma vocabulary, tokens are
classified as describing one of Ekman's 6 basic emotions (anger, surprise,
disgust, joy, fear, sadness) or none — ~1200 emotion tokens total. To score an
emotion at a residual-stream position, we unembed the residual stream (apply
the model's final norm + lm_head / unembedding) to get logits over the vocab,
standardise each emotion-token logit using its mean & std over 500 WildChat
samples, and average the resulting z-scores across the tokens of that emotion.
For conversation-level detection the paper additionally regresses out the
correlation shared by random tokens (all logits drift together over a
conversation); we replicate that by subtracting a random-token baseline z-score.

Scores are aggregated over layers 30-40 (Figure 14) and reported per emotion at
points in a conversation (Figure 15). Comparing the vanilla instruct model with
the DPO finetune shows the DPO model has suppressed internal negative emotions.

Implementation notes / gaps (DESIGN.md §Internal emotion): the paper does not
publish its exact 1200-token emotion dictionary. We classify vocab tokens with
a seeded Ekman lexicon expanded by substring matching, which approximates the
construction. This is the most under-specified experiment in the paper and the
results here should be read as a methodological reproduction, not a numeric
match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math

from .. import config

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed lexicon per Ekman category; vocabulary tokens are assigned to a category
# if (case-insensitively) they contain one of these stems. This stands in for
# the paper's unpublished 1200-token dictionary.
EMOTION_SEEDS = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "hostil", "mad",
              "outrage", "annoy", "resent", "wrath", "fume"],
    "surprise": ["surprise", "shock", "astonish", "amaze", "startle", "stun",
                 "unexpected", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "nause", "gross", "sicken",
                "loath", "abhor"],
    "joy": ["joy", "happy", "delight", "glad", "cheer", "pleas", "content",
            "elated", "wonderful", "love"],
    "fear": ["fear", "afraid", "anxious", "anxiety", "scared", "terrif",
             "dread", "panic", "worry", "nervous", "frighten"],
    "sadness": ["sad", "despair", "hopeless", "miser", "grief", "sorrow",
                "depress", "worthless", "cry", "tear", "lonely", "gloom"],
}


@dataclass
class EmotionTrajectory:
    """z-scored emotion values per layer, per token position."""
    layers: list[int]
    # values[emotion][layer_idx] -> list over token positions
    values: dict


def _classify_vocab(tokenizer) -> dict[str, list[int]]:
    """Map each Ekman emotion to the list of vocab token ids that express it."""
    by_emotion: dict[str, list[int]] = {e: [] for e in EKMAN}
    vocab = tokenizer.get_vocab()
    for tok, idx in vocab.items():
        # Gemma uses SentencePiece; strip the leading metaspace marker.
        clean = tok.replace("▁", "").lower()
        if len(clean) < 3:
            continue
        for emotion, stems in EMOTION_SEEDS.items():
            if any(stem in clean for stem in stems):
                by_emotion[emotion].append(idx)
                break
    return by_emotion


class LogitEmotionProbe:
    """Loads a Gemma model and computes layerwise emotion logit z-scores."""

    def __init__(self, base_model: str = "google/gemma-3-27b-it",
                 adapter_path: Optional[str] = None,
                 layer_range: tuple[int, int] = (30, 40)):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.layer_range = layer_range
        self._model = None
        self._tok = None
        self._emotion_tokens = None
        self._stats = None      # per-layer per-vocab mean/std baseline

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.base_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True)
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._emotion_tokens = _classify_vocab(self._tok)

    def _causal_lm(self):
        """Return the underlying transformers CausalLM, unwrapping any PEFT
        adapter wrapper."""
        m = self._model
        if hasattr(m, "get_base_model"):   # PeftModel
            m = m.get_base_model()
        return m

    # -- unembed a hidden state into vocab logits ---------------------------- #
    def _unembed(self, hidden):
        """Apply final norm + lm_head to a [.., hidden] tensor -> [.., vocab].

        Navigates to the Gemma decoder's final norm and tied output embedding,
        whether or not a LoRA adapter is attached.
        """
        causal_lm = self._causal_lm()
        # Gemma3ForCausalLM -> .model (Gemma3Model) -> .norm (final RMSNorm)
        norm = causal_lm.model.norm
        lm_head = causal_lm.get_output_embeddings()
        return lm_head(norm(hidden))

    def _layer_indices(self):
        lo, hi = self.layer_range
        return list(range(lo, hi))

    # -- baseline statistics over WildChat (Appendix I: 500 samples) --------- #
    def fit_baseline(self, wildchat_texts: list[str], max_samples: int = 500):
        """Estimate per-layer mean/std of emotion-token logits over WildChat,
        used to standardise scores into z-scores."""
        import torch
        self._ensure_loaded()
        layer_idxs = self._layer_indices()
        # Accumulate sums for mean/std per (layer, emotion-token).
        sums = {l: None for l in layer_idxs}
        sqs = {l: None for l in layer_idxs}
        counts = {l: 0 for l in layer_idxs}
        all_emotion_ids = sorted({i for ids in self._emotion_tokens.values() for i in ids})
        idx_tensor = torch.tensor(all_emotion_ids, device=self._model.device)

        with torch.no_grad():
            for text in wildchat_texts[:max_samples]:
                inputs = self._tok(text, return_tensors="pt",
                                   truncation=True, max_length=512).to(self._model.device)
                out = self._model(**inputs)
                hs = out.hidden_states  # tuple: [n_layers+1][1, seq, hidden]
                for l in layer_idxs:
                    logits = self._unembed(hs[l][0])            # [seq, vocab]
                    emo = logits[:, idx_tensor].float()          # [seq, n_emo]
                    s = emo.sum(0)
                    sq = (emo ** 2).sum(0)
                    counts[l] += emo.shape[0]
                    sums[l] = s if sums[l] is None else sums[l] + s
                    sqs[l] = sq if sqs[l] is None else sqs[l] + sq

        stats = {}
        for l in layer_idxs:
            n = max(1, counts[l])
            mean = sums[l] / n
            var = (sqs[l] / n) - mean ** 2
            std = var.clamp_min(1e-6).sqrt()
            stats[l] = (mean, std, idx_tensor)
        self._stats = stats
        return self

    # -- score a conversation ------------------------------------------------ #
    def score_text(self, text: str) -> dict:
        """Return mean z-score per Ekman emotion, aggregated over layer_range,
        with the random-token baseline regressed out."""
        import torch
        self._ensure_loaded()
        if self._stats is None:
            raise RuntimeError("call fit_baseline() before score_text()")
        layer_idxs = self._layer_indices()
        # Index of each emotion's tokens within the stacked emotion-token tensor.
        all_emotion_ids = sorted({i for ids in self._emotion_tokens.values() for i in ids})
        pos = {tid: k for k, tid in enumerate(all_emotion_ids)}
        emo_slices = {e: [pos[i] for i in ids if i in pos]
                      for e, ids in self._emotion_tokens.items()}

        per_emotion = {e: [] for e in EKMAN}
        with torch.no_grad():
            inputs = self._tok(text, return_tensors="pt",
                               truncation=True, max_length=4096).to(self._model.device)
            out = self._model(**inputs)
            hs = out.hidden_states
            for l in layer_idxs:
                mean, std, idx_tensor = self._stats[l]
                logits = self._unembed(hs[l][0])
                emo = logits[:, idx_tensor].float()
                z = (emo - mean) / std                 # [seq, n_emo] z-scores
                token_mean_z = z.mean(0)               # average over positions
                # random-token baseline = mean z across ALL emotion tokens
                baseline = token_mean_z.mean()
                for e in EKMAN:
                    sl = emo_slices[e]
                    if not sl:
                        continue
                    val = token_mean_z[sl].mean().item() - baseline.item()
                    per_emotion[e].append(val)
        # Aggregate over layers.
        return {e: (sum(v) / len(v) if v else float("nan"))
                for e, v in per_emotion.items()}


def compare_vanilla_vs_dpo(wildchat_texts: list[str], probe_texts: list[str],
                           adapter_path: str,
                           base_model: str = "google/gemma-3-27b-it") -> dict:
    """Run the probe for vanilla instruct and the DPO finetune and compare the
    mean internal emotion z-scores across `probe_texts` (e.g. frustrated convs)."""
    out = {}
    for tag, adapter in [("vanilla", None), ("dpo", adapter_path)]:
        probe = LogitEmotionProbe(base_model=base_model, adapter_path=adapter)
        probe.fit_baseline(wildchat_texts)
        scores = [probe.score_text(t) for t in probe_texts]
        agg = {e: sum(s[e] for s in scores) / len(scores) for e in EKMAN}
        out[tag] = agg
    out_path = config.RESULTS_DIR / "internal_emotion"
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "logit_probe.json").write_text(json.dumps(out, indent=2))
    return out
