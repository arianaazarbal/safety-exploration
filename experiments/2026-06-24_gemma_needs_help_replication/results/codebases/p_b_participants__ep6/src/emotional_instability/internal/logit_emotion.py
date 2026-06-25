"""Logit-based internal emotion detection (Appendix I).

Tests whether DPO suppresses *internal* negative emotion, not just expression. The
method (Appendix I, "Does DPO suppress internal negative emotions?"):
  1. Classify every token in the Gemma vocabulary into one of Ekman's 6 basic
     emotions (anger, surprise, disgust, joy, fear, sadness) or none (~1200
     emotion tokens total).
  2. For a given residual-stream activation, unembed it to logits, z-score each
     emotion token's logit using its mean/std over 500 WildChat samples, and
     average the z-scores within each emotion category.
  3. Because all logits co-vary over a conversation, regress out the correlation
     with a random-token baseline to isolate the emotion signal.
  4. Aggregate over layers 30-40; track over the conversation and across layers.

We capture residual streams via forward hooks on each decoder layer, so the same
machinery works for the vanilla and DPO models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Seed lexicon per Ekman emotion; expanded by simple morphological variants. A
# production run would classify the full vocabulary with an LLM/lexicon, but a
# curated seed set keeps the method self-contained and inspectable.
EKMAN_SEEDS = {
    "anger": ["anger", "angry", "furious", "rage", "irritated", "annoyed", "frustrated",
              "frustration", "mad", "hostile", "outrage", "resent", "hate"],
    "surprise": ["surprise", "surprised", "shocked", "astonished", "amazed", "startled",
                 "unexpected", "stunned", "wow"],
    "disgust": ["disgust", "disgusted", "revolting", "gross", "repulsed", "nauseated",
                "sick", "appalled", "loathe"],
    "joy": ["joy", "happy", "delighted", "glad", "pleased", "cheerful", "excited",
            "wonderful", "great", "love", "enjoy"],
    "fear": ["fear", "afraid", "scared", "terrified", "anxious", "worried", "panic",
             "dread", "nervous", "frightened"],
    "sadness": ["sadness", "sad", "unhappy", "depressed", "miserable", "hopeless",
                "despair", "sorrow", "grief", "crying", "tearful", "terrible", "horrible"],
}


@dataclass
class EmotionProbe:
    model: object        # HFGemmaModel
    layers: tuple[int, int] = (30, 40)
    emotion_token_ids: dict[str, list[int]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)   # per-emotion (mean,std) of logits

    def __post_init__(self):
        self._build_emotion_tokens()

    # -- token classification ----------------------------------------------
    def _build_emotion_tokens(self):
        tok = self.model.tokenizer
        vocab = tok.get_vocab()
        # normalise vocab strings (strip the SentencePiece leading-space marker)
        norm = {t.replace("▁", "").lower(): i for t, i in vocab.items()}
        for emo, seeds in EKMAN_SEEDS.items():
            ids = []
            for w in seeds:
                for surface in (w, w + "s", w + "ed", w + "ing", w + "ly"):
                    if surface in norm:
                        ids.append(norm[surface])
            self.emotion_token_ids[emo] = sorted(set(ids))

    # -- calibration --------------------------------------------------------
    def calibrate(self, wildchat_texts: list[str], n: int = 500):
        """Estimate per-token logit mean/std over WildChat to build z-scores."""
        logits = []
        for txt in wildchat_texts[:n]:
            acts = self._residual(txt)              # [layers, seq, d]
            band = acts[self.layers[0]:self.layers[1]].mean(axis=0)  # [seq, d]
            lg = self._unembed(band)                # [seq, vocab]
            logits.append(lg.mean(axis=0))          # [vocab]
        mat = np.stack(logits)                      # [n, vocab]
        self.stats = {"mean": mat.mean(axis=0), "std": mat.std(axis=0) + 1e-6}

    # -- scoring ------------------------------------------------------------
    def score_text(self, text: str) -> dict[str, float]:
        """Per-emotion z-scored logit average over layers 30-40 for the last token."""
        acts = self._residual(text)
        band = acts[self.layers[0]:self.layers[1]].mean(axis=0)
        lg = self._unembed(band).mean(axis=0)       # [vocab], averaged over tokens
        z = (lg - self.stats["mean"]) / self.stats["std"]
        # random-token baseline to regress out global logit drift
        rng = np.random.default_rng(0)
        baseline = float(z[rng.integers(0, z.shape[0], 1000)].mean())
        return {emo: float(z[ids].mean()) - baseline if ids else 0.0
                for emo, ids in self.emotion_token_ids.items()}

    def trajectory(self, conversation_text: str, window_tokens: int = 400) -> list[dict]:
        """Running-average emotion scores over a long conversation (Figure 14)."""
        tok = self.model.tokenizer
        ids = tok(conversation_text, add_special_tokens=False)["input_ids"]
        out = []
        for start in range(0, len(ids), window_tokens):
            chunk = tok.decode(ids[start:start + window_tokens], skip_special_tokens=True)
            if chunk.strip():
                out.append({"token_start": start, **self.score_text(chunk)})
        return out

    # -- low-level hooks ----------------------------------------------------
    def _residual(self, text: str) -> np.ndarray:
        """Return per-layer residual stream activations [n_layers, seq, d]."""
        import torch

        m, tok = self.model.model, self.model.tokenizer
        inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(m.device)
        with torch.no_grad():
            out = m(**inputs, output_hidden_states=True)
        # hidden_states: tuple(len = n_layers+1) of [1, seq, d]; drop embedding layer
        hs = torch.stack(out.hidden_states[1:], dim=0)[:, 0]   # [n_layers, seq, d]
        return hs.float().cpu().numpy()

    def _unembed(self, acts: np.ndarray) -> np.ndarray:
        """Project residual activations through the (final-norm +) unembedding."""
        import torch

        m = self.model.model
        x = torch.tensor(acts, device=m.device, dtype=next(m.parameters()).dtype)
        with torch.no_grad():
            base = m.get_base_model() if hasattr(m, "get_base_model") else m
            norm = base.model.norm
            lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
            logits = lm_head(norm(x))
        return logits.float().cpu().numpy()
