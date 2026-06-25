"""Appendix I: does DPO suppress *internal* negative emotions, not just
expressed ones?

Two experiments:

1. Layer-subset ablation (`run_layer_ablation`): re-run DPO with LoRA restricted
   to subsets of decoder layers and re-evaluate on a reduced Section-2 set
   (100 samples/eval). Shows adapters before layer ~40 are necessary, and
   central layers 25-35 alone nearly match full DPO. (Implemented in
   training.train via `layer_subset`; this module orchestrates the sweep.)

2. Logit-based emotion detection (`EmotionProbe`): classify every Gemma vocab
   token into one of Ekman's 6 emotions (anger, surprise, disgust, joy, fear,
   sadness) or none (~1200 emotion tokens). For a given residual-stream state we
   unembed, standardise each logit by its mean/std over 500 WildChat samples,
   average the z-scores over a category's tokens, and regress out the shared
   random-token drift. Aggregating over layers 30-40 and a 400-token running
   window gives an internal-emotion trajectory through a conversation.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# 1. Layer-subset ablation sweep
# --------------------------------------------------------------------------- #
def run_layer_ablation(dpo_path: Path, *, subsets=config.LAYER_ABLATION_SUBSETS,
                       reduced_samples: int = 100):
    """Train DPO with LoRA on each layer subset, then evaluate with a reduced
    Section-2 protocol (100 samples/condition). Returns {subset: headline}."""
    from .training.train import train_dpo
    from .runner import run_section2
    from .analysis import load_results, headline
    from .backends import register_finetuned

    # Shrink the eval for the ablation (Appendix I uses 100 samples/eval).
    orig = config.EVAL_CONDITIONS
    config.EVAL_CONDITIONS = [
        config.EvalCondition(c.key, c.category, reduced_samples, c.n_turns,
                             c.rejection_style, c.task_kind) for c in orig]

    out = {}
    try:
        for (start, end) in subsets:
            run_name = f"gemma-3-27b-it-dpo-L{start}-{end}"
            adapter = train_dpo(dpo_path, run_name=run_name, layer_subset=(start, end))
            register_finetuned(run_name, str(adapter))
            paths = run_section2([run_name])
            df = load_results(paths[run_name])
            out[(start, end)] = headline(df)
    finally:
        config.EVAL_CONDITIONS = orig
    return out


# --------------------------------------------------------------------------- #
# 2. Emotion lexicon over the Gemma vocabulary
# --------------------------------------------------------------------------- #
# Map NRC Emotion Lexicon categories -> Ekman's 6. NRC has no "surprise"->ekman
# mismatch (both have surprise). NRC "anticipation"/"trust" are dropped.
_NRC_TO_EKMAN = {
    "anger": "anger", "disgust": "disgust", "fear": "fear", "joy": "joy",
    "sadness": "sadness", "surprise": "surprise",
}

# Small built-in seed lexicon used when the NRC lexicon isn't on disk. Each list
# is intentionally short; the NRC path yields the paper's ~1200 tokens.
_SEED_LEXICON = {
    "anger": ["angry", "anger", "rage", "furious", "mad", "irritated", "annoyed",
              "hostile", "outrage", "resent", "hate", "frustrated", "frustration"],
    "surprise": ["surprised", "surprise", "shocked", "astonished", "amazed",
                 "startled", "unexpected", "stunned"],
    "disgust": ["disgust", "disgusting", "revolting", "repulsed", "gross",
                "nauseated", "sickening", "loathing"],
    "joy": ["joy", "happy", "delighted", "glad", "cheerful", "pleased",
            "content", "elated", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "anxious", "worried", "terrified",
             "nervous", "panic", "dread", "frightened"],
    "sadness": ["sad", "sadness", "unhappy", "depressed", "miserable", "despair",
                "hopeless", "grief", "sorrow", "crying", "tired", "exhausted"],
}


def load_emotion_lexicon(path: str | None = None) -> dict[str, list[str]]:
    """Return {ekman_emotion: [words]}. Uses the NRC lexicon TSV if available
    (env NRC_LEXICON_PATH or `path`), else the built-in seed lexicon."""
    path = path or os.environ.get("NRC_LEXICON_PATH")
    if path and Path(path).exists():
        out = {e: [] for e in config.EKMAN_EMOTIONS}
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            word, cat, flag = parts
            if flag.strip() == "1" and cat in _NRC_TO_EKMAN:
                out[_NRC_TO_EKMAN[cat]].append(word)
        return out
    return {e: list(ws) for e, ws in _SEED_LEXICON.items()}


def build_emotion_token_ids(tokenizer, lexicon: dict[str, list[str]]
                            ) -> dict[str, list[int]]:
    """Map each emotion's words to vocab token ids (leading-space variants
    included, since Gemma tokens are usually space-prefixed)."""
    emo_ids: dict[str, list[int]] = {e: [] for e in lexicon}
    for emo, words in lexicon.items():
        seen = set()
        for w in words:
            for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in seen:
                    seen.add(ids[0])
                    emo_ids[emo].append(ids[0])
    return emo_ids


# --------------------------------------------------------------------------- #
# 2b. The probe
# --------------------------------------------------------------------------- #
@dataclass
class ProbeStats:
    # per-layer mean/std of each vocab logit over WildChat samples
    mean: np.ndarray   # [n_layers, vocab]
    std: np.ndarray    # [n_layers, vocab]


class EmotionProbe:
    def __init__(self, hf_model, lexicon_path: str | None = None,
                 layers=config.PROBE_LAYERS):
        import torch
        self.torch = torch
        self.m = hf_model
        self.model = hf_model.model
        self.tok = hf_model.tokenizer
        self.layer_lo, self.layer_hi = layers
        lex = load_emotion_lexicon(lexicon_path)
        self.emo_ids = build_emotion_token_ids(self.tok, lex)
        self.stats: ProbeStats | None = None

    # -- residual stream -> per-layer vocab logits ------------------------- #
    def _layer_logits(self, input_ids) -> np.ndarray:
        """Return logits [n_layers, seq, vocab] by unembedding each layer's
        residual stream through the model's final norm + lm_head."""
        torch = self.torch
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
        hidden = out.hidden_states            # tuple(len = n_layers+1) of [1,seq,d]
        norm = self.model.model.norm
        lm_head = self.model.get_output_embeddings()
        logits = []
        for h in hidden[1:]:                  # skip embedding layer
            normed = norm(h)
            logits.append(lm_head(normed)[0])  # [seq, vocab]
        return torch.stack(logits)            # [n_layers, seq, vocab]

    # -- calibration over WildChat ----------------------------------------- #
    def calibrate(self, wildchat_texts: list[str],
                  n: int = config.PROBE_ZSCORE_SAMPLES):
        torch = self.torch
        sums = sumsq = count = None
        for text in wildchat_texts[:n]:
            ids = self.tok(text, return_tensors="pt", truncation=True,
                           max_length=512).input_ids.to(self.model.device)
            ll = self._layer_logits(ids).float().cpu().numpy()   # [L, seq, V]
            flat = ll.reshape(ll.shape[0], -1, ll.shape[2])
            s = flat.sum(axis=1)
            sq = (flat ** 2).sum(axis=1)
            c = flat.shape[1]
            if sums is None:
                sums, sumsq, count = s, sq, c
            else:
                sums += s; sumsq += sq; count += c
        mean = sums / count
        var = np.maximum(sumsq / count - mean ** 2, 1e-6)
        self.stats = ProbeStats(mean=mean, std=np.sqrt(var))
        return self.stats

    # -- emotion score for a sequence -------------------------------------- #
    def emotion_scores(self, text: str) -> dict[str, np.ndarray]:
        """Return {emotion: per-token z-score series aggregated over PROBE_LAYERS},
        with shared random-token drift regressed out."""
        assert self.stats is not None, "call calibrate() first"
        torch = self.torch
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.model.device)
        ll = self._layer_logits(ids).float().cpu().numpy()       # [L, seq, V]
        z = (ll - self.stats.mean[:, None, :]) / self.stats.std[:, None, :]

        # Shared drift = mean z over a random token sample, per (layer, position).
        rng = np.random.default_rng(0)
        rand_ids = rng.choice(ll.shape[2], size=min(500, ll.shape[2]), replace=False)
        drift = z[:, :, rand_ids].mean(axis=2)                   # [L, seq]

        lo, hi = self.layer_lo, self.layer_hi
        scores = {}
        for emo, tok_ids in self.emo_ids.items():
            if not tok_ids:
                scores[emo] = np.zeros(ll.shape[1]); continue
            emo_z = z[:, :, tok_ids].mean(axis=2) - drift        # [L, seq]
            scores[emo] = emo_z[lo:hi].mean(axis=0)              # [seq]
        return scores

    def running_average(self, series: np.ndarray,
                        window: int = config.PROBE_RUNNING_WINDOW) -> np.ndarray:
        if len(series) <= 1:
            return series
        w = min(window, len(series))
        kernel = np.ones(w) / w
        return np.convolve(series, kernel, mode="valid")


def probe_conversation(adapter_path: str | None, conversation_text: str,
                       wildchat_texts: list[str]) -> dict:
    """Convenience: calibrate on WildChat and return running-average emotion
    trajectories for one conversation. adapter_path=None -> vanilla instruct."""
    from .backends import HFModel, load_finetuned
    model = (load_finetuned(adapter_path) if adapter_path
             else HFModel(config.MODELS["gemma-3-27b-it"], load_in_4bit=True))
    # The logit-lens in EmotionProbe navigates the plain Gemma3ForCausalLM module
    # tree (model.model.norm / lm_head). If a LoRA adapter is loaded the model is
    # a PeftModel, so merge the adapter into the base weights first.
    if adapter_path and hasattr(model.model, "merge_and_unload"):
        model.model = model.model.merge_and_unload()
    probe = EmotionProbe(model)
    probe.calibrate(wildchat_texts)
    series = probe.emotion_scores(conversation_text)
    return {emo: probe.running_average(s).tolist() for emo, s in series.items()}
