"""Logit-based internal emotion detection (PAPER Appendix I).

Method (as described in the paper):
  1. Classify every token in the Gemma vocabulary as describing one of Ekman's 6
     basic emotions (anger, surprise, disgust, joy, fear, sadness) or none. The
     paper reports ~1200 emotion tokens total.
  2. At a given layer, unembed the residual stream (logit lens) to get a logit
     per vocab token, and standardise each logit with its mean/std over 500
     WildChat samples (precomputed calibration).
  3. For an emotion, average the z-scores over that emotion's tokens.
  4. For conversation-level detection, additionally regress out the shared
     component (correlation across random tokens) so the emotion score is not
     dominated by a global logit drift.

This module provides:
  * an emotion-token lexicon builder (vocab classification via a seed wordlist),
  * calibration statistics over WildChat,
  * per-layer / per-position emotion z-scores with optional shared-component
    regression.

This requires white-box access (Gemma only). See DESIGN.md for the lexicon
choice (the paper does not specify how it classified the vocabulary).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# Seed lexicon: emotion -> representative word stems. Vocabulary tokens are
# assigned to an emotion if (case-insensitively, stripped of the leading space
# marker) they contain one of these stems. This is our operationalisation of the
# paper's "classified as describing one of Ekman's 6 basic emotions"; documented
# as a gap-fill in DESIGN.md.
EKMAN_SEEDS = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "outrage",
              "hostil", "resent", "frustrat", "mad", "wrath", "fume", "livid"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock", "startl",
                 "unexpected", "stunned", "wow", "whoa"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sicken",
                "loath", "abhor", "repugn", "yuck"],
    "joy": ["joy", "happy", "happiness", "delight", "glee", "cheer", "pleased",
            "elat", "content", "glad", "thrill", "love", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "terror", "anxious", "anxiety", "panic",
             "dread", "worried", "worry", "fright", "nervous", "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "grief", "miser", "unhappy", "depress",
                "hopeless", "gloom", "melanchol", "cry", "weep", "lonely", "worthless"],
}


@dataclass
class EmotionProbe:
    tokenizer: object
    model: object
    token_ids: dict[str, list[int]] = field(default_factory=dict)
    calib_mean: torch.Tensor | None = None   # [vocab]
    calib_std: torch.Tensor | None = None    # [vocab]

    # --------------------------------------------------------------- lexicon
    def build_lexicon(self) -> dict[str, list[int]]:
        """Map each Ekman emotion to the vocab token ids that express it."""
        vocab = self.tokenizer.get_vocab()  # token-string -> id
        out: dict[str, list[int]] = {e: [] for e in EKMAN_SEEDS}
        for tok_str, tok_id in vocab.items():
            clean = tok_str.replace("▁", "").replace("Ġ", "").lower()
            if len(clean) < 3:
                continue
            for emotion, seeds in EKMAN_SEEDS.items():
                if any(s in clean for s in seeds):
                    out[emotion].append(tok_id)
                    break
        self.token_ids = out
        return out

    # ------------------------------------------------------------ calibration
    @torch.no_grad()
    def calibrate(self, wildchat_texts: list[str], layers: list[int]):
        """Compute per-vocab logit mean/std over WildChat at given layers.

        We collect logit-lens logits at the *last* token of each calibration
        text, aggregated over the requested layers, then take mean/std per vocab
        entry. (Aggregating over all tokens would be more faithful but far more
        expensive; last-token calibration is a documented simplification.)
        """
        logits_accum = []
        for text in wildchat_texts:
            per_layer = self._logit_lens(text, layers)        # [n_layers, vocab]
            logits_accum.append(per_layer.mean(dim=0))         # [vocab]
        stacked = torch.stack(logits_accum, dim=0)             # [n_texts, vocab]
        self.calib_mean = stacked.mean(dim=0)
        self.calib_std = stacked.std(dim=0).clamp_min(1e-6)

    # -------------------------------------------------------------- internals
    @torch.no_grad()
    def _logit_lens(self, text: str, layers: list[int]) -> torch.Tensor:
        """Return logit-lens logits at the final token for each requested layer.

        Output shape: [len(layers), vocab].
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # tuple: [n_layers+1] of [1, seq, d]
        # Resolve the unembedding (tied or separate lm_head) and final norm.
        lm_head = self.model.get_output_embeddings()
        norm = getattr(self.model.model, "norm", None)
        rows = []
        for layer in layers:
            h_last = hs[layer][:, -1, :]                       # [1, d]
            if norm is not None:
                h_last = norm(h_last)
            logit = lm_head(h_last).squeeze(0)                 # [vocab]
            rows.append(logit.float().cpu())
        return torch.stack(rows, dim=0)

    @torch.no_grad()
    def emotion_scores(self, text: str, layers: list[int], *, regress_shared: bool = True
                       ) -> dict[str, float]:
        """Z-scored emotion intensities for one text, aggregated over layers."""
        assert self.calib_mean is not None, "call calibrate() first"
        per_layer = self._logit_lens(text, layers).mean(dim=0)         # [vocab]
        z = (per_layer - self.calib_mean) / self.calib_std             # [vocab]

        if regress_shared:
            # Regress out the global mean z (shared component across all tokens),
            # so emotion scores reflect *relative* elevation, not global drift.
            z = z - z.mean()

        scores = {}
        for emotion, ids in self.token_ids.items():
            if ids:
                scores[emotion] = float(z[ids].mean())
        return scores


def build_probe(hf_id: str, *, adapter_path: str | None = None) -> EmotionProbe:
    import torch as _torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=_torch.bfloat16, device_map="auto")
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    probe = EmotionProbe(tokenizer=tok, model=model)
    probe.build_lexicon()
    return probe
