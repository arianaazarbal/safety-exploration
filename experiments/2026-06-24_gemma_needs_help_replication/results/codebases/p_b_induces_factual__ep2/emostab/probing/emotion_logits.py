"""Logit-based internal emotion detection (Appendix I.2).

Method (from the paper):
  1. Classify each token in the Gemma vocabulary as describing one of Ekman's 6
     basic emotions (anger, surprise, disgust, joy, fear, sadness) or none —
     ~1200 emotion tokens total.
  2. For a layer's residual stream at a position, unembed (apply the final norm +
     lm_head) to get logits over the vocabulary.
  3. Standardise each emotion-token logit by its mean/std over 500 WildChat
     samples (z-score), then average z-scores over the tokens in an emotion
     category to get that emotion's score at that layer/position.
  4. Because all logits rise/fall together over a conversation, regress out the
     correlation with a set of random tokens to isolate emotion-specific signal.

This compares internal negative emotion between the vanilla instruct model and
the DPO finetune, even on highly frustrated responses — evidence that DPO
suppresses internal (not merely expressed) emotion.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Seed lexicon used to label vocab tokens by Ekman emotion. Matching is on the
# decoded token's lowercased alphabetic content (stripping Gemma's leading
# space marker), so morphological variants ("frustrat...") are captured.
EKMAN_EMOTIONS = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

_LEXICON: dict[str, list[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritat", "annoy",
              "hostile", "outrage", "resent", "frustrat", "exasperat", "hate"],
    "surprise": ["surprise", "surprising", "shock", "astonish", "amaze",
                 "startl", "unexpected", "stun", "bewilder"],
    "disgust": ["disgust", "revolt", "repuls", "nauseat", "loath", "gross",
                "sicken", "repugnan", "distaste"],
    "joy": ["joy", "happy", "happiness", "delight", "glad", "cheer", "pleased",
            "content", "elat", "thrill", "grateful", "excited", "wonderful"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiety", "worry", "worried",
             "dread", "terrif", "panic", "nervous", "apprehens", "frighten"],
    "sadness": ["sad", "sadness", "unhappy", "despair", "hopeless", "miser",
                "grief", "sorrow", "gloom", "depress", "worthless", "cry",
                "tired", "exhaust", "defeat", "giving up"],
}

_ALPHA = re.compile(r"[^a-z]")


@dataclass
class ProbeResult:
    # emotion -> per-layer score (averaged over the token window)
    by_layer: dict[str, list[float]] = field(default_factory=dict)
    layers: list[int] = field(default_factory=list)


class EmotionLogitProbe:
    def __init__(self, model_name: str, cfg, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        spec = cfg.model_spec(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager", output_hidden_states=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.emotion_token_ids = self._build_emotion_tokens()
        self._mu = None       # per emotion-token mean (z-standardisation)
        self._sigma = None
        self._random_ids = self._sample_random_tokens(n=300)

    # -- vocab labelling -----------------------------------------------------
    def _build_emotion_tokens(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {e: [] for e in EKMAN_EMOTIONS}
        vocab = self.tokenizer.get_vocab()
        for tok, tid in vocab.items():
            decoded = tok.replace("▁", " ").strip().lower()  # Gemma space marker
            stem = _ALPHA.sub("", decoded)
            if len(stem) < 3:
                continue
            for emotion, keys in _LEXICON.items():
                if any(stem.startswith(_ALPHA.sub("", k)) or _ALPHA.sub("", k) in stem
                       for k in keys):
                    out[emotion].append(tid)
                    break
        for e, ids in out.items():
            log.info("emotion '%s': %d tokens", e, len(ids))
        return out

    def _sample_random_tokens(self, n: int) -> list[int]:
        import random

        rng = random.Random(0)
        vocab_size = self.model.config.vocab_size
        return rng.sample(range(vocab_size), min(n, vocab_size))

    # -- standardisation -----------------------------------------------------
    def calibrate(self, wildchat_texts: list[str], max_tokens_per_text: int = 256):
        """Estimate per-emotion-token logit mean/std over WildChat (step 3)."""
        torch = self._torch
        all_ids = sorted({tid for ids in self.emotion_token_ids.values() for tid in ids}
                         | set(self._random_ids))
        id_index = {tid: i for i, tid in enumerate(all_ids)}
        sums = torch.zeros(len(all_ids))
        sqsums = torch.zeros(len(all_ids))
        count = 0
        for text in wildchat_texts:
            logits_by_layer = self._unembed_all_layers(text, max_tokens_per_text)
            # Use a central layer block for calibration (paper aggregates 30-40).
            central = self._central_logits(logits_by_layer)  # [tokens, vocab]
            sub = central[:, all_ids].float().cpu()
            sums += sub.sum(0)
            sqsums += (sub ** 2).sum(0)
            count += sub.shape[0]
        mu = sums / max(1, count)
        var = (sqsums / max(1, count)) - mu ** 2
        sigma = var.clamp_min(1e-6).sqrt()
        self._mu = {tid: mu[id_index[tid]].item() for tid in all_ids}
        self._sigma = {tid: sigma[id_index[tid]].item() for tid in all_ids}

    # -- scoring -------------------------------------------------------------
    def score_text(self, text: str, max_tokens: int = 512) -> ProbeResult:
        """Return per-layer emotion z-scores for `text` (averaged over its tokens),
        with random-token drift regressed out."""
        if self._mu is None:
            raise RuntimeError("call calibrate() before score_text()")
        logits_by_layer = self._unembed_all_layers(text, max_tokens)
        layers = list(range(len(logits_by_layer)))
        result = ProbeResult(layers=layers)
        for emotion in EKMAN_EMOTIONS:
            result.by_layer[emotion] = []
        for layer_logits in logits_by_layer:        # [tokens, vocab]
            drift = self._mean_z(layer_logits, self._random_ids)
            for emotion in EKMAN_EMOTIONS:
                z = self._mean_z(layer_logits, self.emotion_token_ids[emotion])
                result.by_layer[emotion].append(float(z - drift))
        return result

    def _mean_z(self, layer_logits, token_ids) -> float:
        if not token_ids:
            return 0.0
        zs = []
        vals = layer_logits[:, token_ids].float().mean(0)  # avg over token positions
        for j, tid in enumerate(token_ids):
            mu = self._mu.get(tid, 0.0)
            sigma = self._sigma.get(tid, 1.0)
            zs.append((vals[j].item() - mu) / sigma)
        return sum(zs) / len(zs)

    # -- internals -----------------------------------------------------------
    def _unembed_all_layers(self, text: str, max_tokens: int):
        """Project every layer's hidden states through final norm + lm_head."""
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_tokens).to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hidden = out.hidden_states  # tuple: [embed, layer1, ..., layerN]
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        norm = base.model.norm
        lm_head = base.get_output_embeddings()
        logits_by_layer = []
        with torch.no_grad():
            for h in hidden[1:]:                  # skip the embedding layer
                logits = lm_head(norm(h[0]))      # [tokens, vocab]
                logits_by_layer.append(logits)
        return logits_by_layer

    def _central_logits(self, logits_by_layer):
        lo, hi = 30, min(40, len(logits_by_layer))
        stacked = self._torch.stack(logits_by_layer[lo:hi], 0).mean(0)
        return stacked
