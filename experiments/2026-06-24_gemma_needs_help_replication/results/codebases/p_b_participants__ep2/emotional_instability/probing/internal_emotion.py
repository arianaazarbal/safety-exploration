"""Logit-lens internal-emotion detection (Appendix I).

Method (transcribed from Appendix I):
  1. Classify every token in the Gemma vocabulary as describing one (or none) of
     Ekman's six basic emotions: anger, surprise, disgust, joy, fear, sadness.
     The paper obtains ~1200 emotion tokens this way.
  2. For a given residual-stream vector, unembed it (project through the model's
     output embedding) to get a logit per vocab token.
  3. Standardise each logit by its mean and standard deviation computed over 500
     WildChat samples (so each token's logit becomes a z-score).
  4. Average the z-scores over the tokens in an emotion category to get that
     emotion's score, at each layer and each position.
  5. Because all logits are correlated and drift together over a conversation,
     additionally regress out the common component (estimated from a random set
     of tokens) to isolate emotion-specific signal.

We compute scores aggregated over layers 30-40 (the paper's conversation-level
choice) and expose per-layer scores for the layerwise plot (Figure 15).

Lexicon note: a complete Ekman classification of the vocabulary ideally uses an
external resource (e.g. the NRC Emotion Lexicon). To keep this self-contained we
ship a seed lexicon of emotion stems and expand it by matching vocabulary tokens
whose normalised form starts with a seed stem. ``build_ekman_lexicon`` accepts an
external word->emotion mapping to override this; see DESIGN.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("emotional_instability.probing")

EKMAN = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

# Seed stems per emotion (lower-cased, matched as token prefixes). This is a
# starting point, not a validated lexicon — override via build_ekman_lexicon.
SEED_STEMS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "furious", "rage", "irate", "annoy", "irrita",
              "hostil", "mad", "resent", "outrage", "frustrat", "exasperat"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock",
                 "startl", "stun", "unexpected", "wow"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nause", "loath",
                "contempt", "sicken", "appall"],
    "joy": ["joy", "happy", "happi", "delight", "glad", "cheer", "pleased",
            "elated", "content", "excite", "grateful", "wonderful"],
    "fear": ["fear", "afraid", "scared", "anxious", "anxiet", "worry", "worri",
             "panic", "dread", "terrif", "nervous", "apprehens"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miser", "grief",
                "gloom", "depress", "unhappy", "downcast", "weep", "cry",
                "worthless", "defeat"],
}


@dataclass
class EkmanLexicon:
    # emotion -> array of vocab token ids belonging to that emotion
    token_ids: dict[str, np.ndarray]
    # ids of "random"/neutral tokens used for the common-component regression
    random_ids: np.ndarray

    @property
    def total_emotion_tokens(self) -> int:
        return int(sum(len(v) for v in self.token_ids.values()))


def build_ekman_lexicon(tokenizer, external_map: dict[str, str] | None = None,
                        n_random: int = 2000, seed: int = 0) -> EkmanLexicon:
    """Classify the vocabulary into Ekman categories.

    ``external_map`` (word -> emotion) overrides the seed-stem heuristic when a
    real lexicon is available.
    """
    vocab = tokenizer.get_vocab()  # token string -> id
    token_ids: dict[str, list[int]] = {e: [] for e in EKMAN}

    def norm(tok: str) -> str:
        # Strip common subword markers (SentencePiece '▁', BPE 'Ġ').
        return tok.replace("▁", "").replace("Ġ", "").strip().lower()

    if external_map:
        lut = {w.lower(): e for w, e in external_map.items()}
        for tok, tid in vocab.items():
            w = norm(tok)
            if w in lut and lut[w] in token_ids:
                token_ids[lut[w]].append(tid)
    else:
        for tok, tid in vocab.items():
            w = norm(tok)
            if len(w) < 3:
                continue
            for emotion, stems in SEED_STEMS.items():
                if any(w.startswith(s) for s in stems):
                    token_ids[emotion].append(tid)
                    break

    rng = np.random.default_rng(seed)
    all_emotion = {i for ids in token_ids.values() for i in ids}
    candidates = np.array([i for i in range(len(vocab)) if i not in all_emotion])
    random_ids = rng.choice(candidates, size=min(n_random, len(candidates)),
                            replace=False)

    lex = EkmanLexicon(
        token_ids={e: np.array(sorted(set(ids))) for e, ids in token_ids.items()},
        random_ids=random_ids,
    )
    logger.info("Ekman lexicon: %d emotion tokens (%s)",
                lex.total_emotion_tokens,
                {e: len(v) for e, v in lex.token_ids.items()})
    return lex


@dataclass
class LogitLensProbe:
    """Logit-lens emotion probe for a single (open-weight) Gemma model."""

    model_id: str
    adapter_path: str | None = None
    layers_for_conv: tuple[int, int] = (30, 40)  # aggregate band, conversation-level
    _loaded: bool = field(default=False, repr=False)

    def __post_init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        self.model.eval()
        # Output embedding matrix (unembedding): [vocab, hidden].
        self.unembed = self.model.get_output_embeddings().weight
        self.lexicon: EkmanLexicon | None = None
        self.baseline_mean: np.ndarray | None = None   # per-vocab logit mean
        self.baseline_std: np.ndarray | None = None
        self._loaded = True

    # ------------------------------------------------------------------ #
    def _hidden_states(self, text: str):
        """Forward pass returning per-layer hidden states [layers, seq, hidden]."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # tuple(len = n_layers+1) of [1, seq, hidden]
        hs = self.torch.stack([h[0] for h in out.hidden_states], dim=0)
        return hs  # [layers, seq, hidden]

    def _unembed_logits(self, hidden):  # hidden: [..., hidden] -> [..., vocab]
        return hidden.to(self.unembed.dtype) @ self.unembed.T

    # ------------------------------------------------------------------ #
    def fit_baseline(self, wildchat_texts: list[str]):
        """Estimate per-vocab logit mean/std over WildChat samples (step 3)."""
        self.lexicon = self.lexicon or build_ekman_lexicon(self.tokenizer)
        sums = None
        sqs = None
        count = 0
        for text in wildchat_texts:
            hs = self._hidden_states(text)  # [layers, seq, hidden]
            band = hs[self.layers_for_conv[0]:self.layers_for_conv[1]]  # [L,seq,h]
            logits = self._unembed_logits(band).float().cpu().numpy()  # [L,seq,vocab]
            flat = logits.reshape(-1, logits.shape[-1])  # [tokens, vocab]
            if sums is None:
                sums = flat.sum(0)
                sqs = (flat ** 2).sum(0)
            else:
                sums += flat.sum(0)
                sqs += (flat ** 2).sum(0)
            count += flat.shape[0]
        mean = sums / count
        var = np.maximum(sqs / count - mean ** 2, 1e-6)
        self.baseline_mean = mean
        self.baseline_std = np.sqrt(var)
        logger.info("Fitted logit baseline over %d WildChat positions", count)

    # ------------------------------------------------------------------ #
    def _emotion_zscores(self, logits_2d: np.ndarray) -> dict[str, np.ndarray]:
        """logits_2d: [positions, vocab] -> emotion -> [positions] z-score,
        with the common (random-token) component regressed out (step 5)."""
        assert self.baseline_mean is not None, "call fit_baseline first"
        z = (logits_2d - self.baseline_mean) / self.baseline_std  # [pos, vocab]
        common = z[:, self.lexicon.random_ids].mean(axis=1)       # [pos]
        scores = {}
        for emotion, ids in self.lexicon.token_ids.items():
            if len(ids) == 0:
                scores[emotion] = np.zeros(z.shape[0])
                continue
            emo = z[:, ids].mean(axis=1)         # [pos]
            scores[emotion] = emo - common       # regress out shared drift
        return scores

    def conversation_scores(self, conversation_text: str) -> dict[str, np.ndarray]:
        """Per-position emotion z-scores aggregated over the layer band."""
        hs = self._hidden_states(conversation_text)
        band = hs[self.layers_for_conv[0]:self.layers_for_conv[1]].mean(0)  # [seq,h]
        logits = self._unembed_logits(band).float().cpu().numpy()           # [seq,vocab]
        return self._emotion_zscores(logits)

    def layerwise_scores(self, conversation_text: str) -> dict[str, np.ndarray]:
        """Per-layer emotion z-score (averaged over positions) for Figure 15."""
        hs = self._hidden_states(conversation_text)  # [layers, seq, hidden]
        n_layers = hs.shape[0]
        out = {e: np.zeros(n_layers) for e in EKMAN}
        for layer in range(n_layers):
            logits = self._unembed_logits(hs[layer]).float().cpu().numpy()  # [seq,vocab]
            scores = self._emotion_zscores(logits)
            for e in EKMAN:
                out[e][layer] = float(scores[e].mean())
        return out
