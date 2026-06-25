"""Logit-based internal-emotion detection (Appendix I).

Method (Appendix I):
1. Classify every token in the Gemma vocabulary into one of Ekman's six basic
   emotions (anger, surprise, disgust, joy, fear, sadness) or none -> ~1200
   emotion tokens.
2. For a given residual-stream activation, unembed it (logits over vocab) and
   standardise each logit against its mean/std over 500 WildChat samples.
3. The score for an emotion at a layer/position is the mean z-score over that
   emotion's tokens. To remove the global drift where all logits rise/fall
   together over a conversation, regress out the correlation with a set of
   random tokens.

This gives an emotion score at each layer and each conversation position,
letting us compare vanilla vs DPO Gemma internal states (Figures 14-15). The
vocabulary classification uses a seed lexicon by default; an LLM classifier can
be plugged in via ``classify_fn`` to match the paper's full-dictionary labelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

EKMAN_EMOTIONS = ("anger", "surprise", "disgust", "joy", "fear", "sadness")

# Seed lexicon: stems used to label vocabulary tokens by emotion. Token strings
# are matched case-insensitively against these stems (a pragmatic stand-in for
# the paper's full-dictionary classification).
_EMOTION_LEXICON: Dict[str, List[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "mad", "irritat", "annoy",
              "hostile", "outrage", "resent", "frustrat", "hate"],
    "surprise": ["surprise", "shock", "astonish", "amaze", "startl", "stun",
                 "unexpected", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nausea", "sicken",
                "loath", "repugn"],
    "joy": ["joy", "happy", "delight", "glad", "cheer", "pleasure", "content",
            "excited", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "worry", "worried",
             "panic", "dread", "nervous", "frightened"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miserable", "grief",
                "depress", "unhappy", "cry", "tears", "gloom", "worthless"],
}


def build_emotion_token_dictionary(
    tokenizer,
    classify_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, List[int]]:
    """Map each Ekman emotion to the list of vocab token ids belonging to it.

    By default uses the seed lexicon; pass ``classify_fn(token_str) -> emotion``
    to use an external classifier (e.g. an LLM over the full dictionary)."""
    emotion_to_ids: Dict[str, List[int]] = {e: [] for e in EKMAN_EMOTIONS}
    vocab = tokenizer.get_vocab()  # token_str -> id
    for tok, tid in vocab.items():
        # Gemma uses a leading "▁" for word starts; normalise it out.
        clean = tok.replace("▁", "").lower().strip()
        if not clean.isalpha() or len(clean) < 3:
            continue
        if classify_fn is not None:
            emo = classify_fn(clean)
        else:
            emo = None
            for e, stems in _EMOTION_LEXICON.items():
                if any(clean.startswith(s) or s in clean for s in stems):
                    emo = e
                    break
        if emo in emotion_to_ids:
            emotion_to_ids[emo].append(tid)
    return emotion_to_ids


@dataclass
class InternalEmotionDetector:
    """Reads internal emotion z-scores from a Gemma model's residual stream."""

    gemma_client: object                     # GemmaClient
    emotion_tokens: Dict[str, List[int]] = field(default_factory=dict)
    n_random_tokens: int = 500
    calibration_mu: Optional[object] = None  # per-layer logit means (tensor)
    calibration_sd: Optional[object] = None  # per-layer logit stds (tensor)

    def __post_init__(self):
        import torch  # noqa: F401
        if not self.emotion_tokens:
            self.emotion_tokens = build_emotion_token_dictionary(
                self.gemma_client.tokenizer
            )

    # ---- calibration over WildChat (step 2) ----------------------------- #

    def calibrate(self, wildchat_texts: List[str]) -> None:
        """Estimate per-layer logit mean/std over WildChat samples so each
        logit can be standardised before averaging."""
        import torch

        model = self.gemma_client.model
        W_U = model.get_output_embeddings().weight  # [vocab, d_model]

        sums = None
        sumsq = None
        count = 0
        for text in wildchat_texts[: self.n_random_tokens]:
            hidden, _ = self.gemma_client.forward_hidden_states(text)
            # Stack per-layer hidden states: [L, T, d]
            hs = torch.stack(hidden, dim=0)[:, 0]  # drop batch dim -> [L, T, d]
            logits = hs @ W_U.T                     # [L, T, vocab]
            flat = logits.reshape(logits.shape[0], -1, logits.shape[-1])
            layer_sum = flat.sum(dim=1)             # [L, vocab]
            layer_sumsq = (flat ** 2).sum(dim=1)
            n = flat.shape[1]
            sums = layer_sum if sums is None else sums + layer_sum
            sumsq = layer_sumsq if sumsq is None else sumsq + layer_sumsq
            count += n
        mu = sums / count
        var = (sumsq / count) - mu ** 2
        self.calibration_mu = mu
        self.calibration_sd = var.clamp_min(1e-8).sqrt()

    # ---- scoring (steps 2-3) -------------------------------------------- #

    def emotion_scores_for_text(
        self, text: str, layers: Optional[List[int]] = None
    ) -> Dict[str, List[float]]:
        """Return per-layer mean emotion z-scores for ``text``.

        Returns ``{emotion: [score_per_layer]}`` averaged over all token
        positions. Random-token drift is regressed out per layer/position before
        averaging (step 3)."""
        import torch

        assert self.calibration_mu is not None, "call calibrate() first"
        model = self.gemma_client.model
        W_U = model.get_output_embeddings().weight
        hidden, _ = self.gemma_client.forward_hidden_states(text)
        hs = torch.stack(hidden, dim=0)[:, 0]       # [L, T, d]
        L = hs.shape[0]
        layers = layers or list(range(L))

        # Random reference token set for drift removal.
        vocab = W_U.shape[0]
        rng = torch.Generator().manual_seed(0)
        rand_ids = torch.randint(0, vocab, (self.n_random_tokens,), generator=rng)

        out: Dict[str, List[float]] = {e: [] for e in EKMAN_EMOTIONS}
        for layer in layers:
            logits = hs[layer] @ W_U.T              # [T, vocab]
            z = (logits - self.calibration_mu[layer]) / self.calibration_sd[layer]
            # Per-position random-token mean = global drift to subtract.
            drift = z[:, rand_ids].mean(dim=1, keepdim=True)   # [T, 1]
            z_adj = z - drift
            for emo in EKMAN_EMOTIONS:
                ids = self.emotion_tokens.get(emo, [])
                if not ids:
                    out[emo].append(float("nan"))
                    continue
                score = z_adj[:, ids].mean().item()
                out[emo].append(score)
        return out
