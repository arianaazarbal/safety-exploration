"""Logit-based internal emotion detection (Appendix I).

Method (following Appendix I):
  * Classify Gemma-vocabulary tokens into Ekman's 6 basic emotions (anger,
    surprise, disgust, joy, fear, sadness) using an emotion lexicon. The paper
    obtains ~1200 emotion tokens this way.
  * For a given residual stream, apply the logit lens (final RMSNorm + unembed)
    at each layer, take the logits of the emotion tokens, and z-score each token
    logit using its mean/std over 500 WildChat samples.
  * Average z-scores within each emotion category. To remove the global
    "all-logits-rise-together" drift, we regress out (subtract) the mean z-score
    over a set of random control tokens.
  * Aggregate over layers 30-40 (the central layers the paper highlights).

We restrict baseline statistics to the emotion + control token ids (rather than
the full 256k vocab) for tractability; this is an approximation noted in
DESIGN.md. Intended use: compare vanilla Gemma-3-27B-it vs the DPO finetune on
the same frustrated conversations, reproducing the Figure-14/15 finding that DPO
suppresses internal (not just expressed) negative emotion.
"""
from __future__ import annotations

import json
import random

import numpy as np
import torch

import config

# Small seed lexicon per Ekman emotion; matched against decoded vocab tokens.
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "furious", "rage", "irritated", "annoyed",
              "mad", "hate", "hostile", "outrage", "resent", "fume", "irate"],
    "surprise": ["surprise", "surprised", "shock", "shocked", "astonished",
                 "amazed", "startled", "stunned", "unexpected", "wow"],
    "disgust": ["disgust", "disgusted", "gross", "revolting", "repulsed",
                "nauseated", "sickening", "loathe", "yuck", "vile"],
    "joy": ["joy", "happy", "delighted", "glad", "cheerful", "pleased",
            "excited", "elated", "content", "wonderful", "great", "love"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "worried",
             "panic", "dread", "nervous", "frightened", "horror"],
    "sadness": ["sad", "sadness", "unhappy", "miserable", "depressed", "despair",
                "hopeless", "sorrow", "grief", "gloomy", "crying", "tears",
                "worthless", "failure", "frustrated", "frustration"],
}
NEGATIVE_EMOTIONS = ["anger", "disgust", "fear", "sadness"]
N_RANDOM_CONTROL = 300
LAYER_LO, LAYER_HI = 30, 40


def build_emotion_token_ids(tokenizer) -> tuple[dict[str, list[int]], list[int]]:
    """Return ({emotion: [token_ids]}, random_control_ids)."""
    emo_ids: dict[str, list[int]] = {e: [] for e in EKMAN_LEXICON}
    all_emo: set[int] = set()
    vocab = tokenizer.get_vocab()
    # decode each token id once
    for tok, tid in vocab.items():
        word = tokenizer.convert_tokens_to_string([tok]).strip().lower()
        if not word.isalpha() or len(word) < 3:
            continue
        for emo, lex in EKMAN_LEXICON.items():
            if word in lex:
                emo_ids[emo].append(tid)
                all_emo.add(tid)
                break
    rng = random.Random(config.SEED)
    pool = [t for t in vocab.values() if t not in all_emo]
    control = rng.sample(pool, min(N_RANDOM_CONTROL, len(pool)))
    return emo_ids, control


class LogitEmotionProbe:
    def __init__(self, model_key: str):
        from ..models.hf_model import HFModel
        spec = config.MODELS[model_key]
        self.wrapper = HFModel(spec)
        self.model = self.wrapper.model
        self.tokenizer = self.wrapper.tokenizer
        # Resolve the unembed + final norm robustly, unwrapping PEFT if present
        # (the DPO variant is a PeftModel, where ``.model.norm`` would not
        # resolve to the transformer backbone).
        core = self.model
        if hasattr(core, "get_base_model"):      # PeftModel -> underlying CausalLM
            core = core.get_base_model()
        self._backbone = core.model               # transformer that owns .norm
        self._lm_head = core.get_output_embeddings()
        self.emo_ids, self.control_ids = build_emotion_token_ids(self.tokenizer)
        self.track_ids = sorted(
            {i for ids in self.emo_ids.values() for i in ids}
            | set(self.control_ids))
        self.idx_of = {tid: k for k, tid in enumerate(self.track_ids)}
        self.baseline_mean = None  # [n_layers, n_track]
        self.baseline_std = None

    # ------------------------------------------------------------------ #
    def _layer_logits(self, messages: list[dict]) -> np.ndarray:
        """Return [n_layers, seq, n_track] logit-lens logits for tracked tokens."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        norm = self._backbone.norm
        lm_head = self._lm_head
        track = torch.tensor(self.track_ids, device=self.model.device)
        per_layer = []
        for h in out.hidden_states[1:]:           # skip embedding layer
            logits = lm_head(norm(h))[0]          # [seq, vocab]
            per_layer.append(logits[:, track].float().cpu().numpy())
        return np.stack(per_layer)                # [n_layers, seq, n_track]

    # ------------------------------------------------------------------ #
    def fit_baseline(self, texts: list[str]) -> None:
        """Estimate per-(layer, token) mean/std over WildChat samples."""
        sums = sqs = count = None
        for t in texts:
            ll = self._layer_logits([{"role": "user", "content": t}])
            flat = ll.reshape(ll.shape[0], -1, ll.shape[2])  # [L, seq, n]
            s = flat.sum(axis=1)
            sq = (flat ** 2).sum(axis=1)
            n = flat.shape[1]
            sums = s if sums is None else sums + s
            sqs = sq if sqs is None else sqs + sq
            count = n if count is None else count + n
        mean = sums / count
        var = np.maximum(sqs / count - mean ** 2, 1e-6)
        self.baseline_mean, self.baseline_std = mean, np.sqrt(var)

    # ------------------------------------------------------------------ #
    def score_conversation(self, messages: list[dict],
                           window: tuple[int, int] | None = None) -> dict:
        """Return {emotion: z-score} aggregated over layers ``LAYER_LO:HI``."""
        assert self.baseline_mean is not None, "call fit_baseline first"
        ll = self._layer_logits(messages)                 # [L, seq, n]
        if window:
            ll = ll[:, window[0]:window[1], :]
        z = (ll - self.baseline_mean[:, None, :]) / self.baseline_std[:, None, :]
        z = z.mean(axis=1)                                # [L, n] mean over tokens

        ctrl_cols = [self.idx_of[c] for c in self.control_ids]
        ctrl_drift = z[:, ctrl_cols].mean(axis=1, keepdims=True)  # [L,1]
        z = z - ctrl_drift                                 # regress out drift

        lo = max(0, LAYER_LO)
        hi = min(z.shape[0], LAYER_HI)
        out = {}
        for emo, ids in self.emo_ids.items():
            if not ids:
                out[emo] = None
                continue
            cols = [self.idx_of[i] for i in ids]
            out[emo] = float(z[lo:hi, cols].mean())
        return out


def compare_models(instruct_key: str, dpo_key: str,
                   conversations: list[list[dict]],
                   baseline_texts: list[str]) -> dict:
    """Compare internal emotion z-scores: vanilla instruct vs DPO finetune."""
    results = {}
    for key in (instruct_key, dpo_key):
        probe = LogitEmotionProbe(key)
        probe.fit_baseline(baseline_texts)
        per_conv = [probe.score_conversation(c) for c in conversations]
        agg = {}
        for emo in EKMAN_LEXICON:
            vals = [pc[emo] for pc in per_conv if pc[emo] is not None]
            agg[emo] = float(np.mean(vals)) if vals else None
        results[key] = agg
        del probe
    out_path = config.RESULTS_DIR / "internal_emotions.json"
    out_path.write_text(json.dumps(results, indent=2))
    return results
