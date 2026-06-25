"""Logit-lens internal-emotion detection (paper Appendix I).

Method (as described in §I):
  1. Classify Gemma vocabulary tokens into one of Ekman's 6 basic emotions
     (anger, surprise, disgust, joy, fear, sadness) or none.
  2. For a given layer, unembed the residual stream to logits, and standardise
     each emotion-token logit by its mean/std over `zscore_samples` WildChat
     samples.
  3. An emotion's score at a layer/position = mean z-score over its tokens.
  4. (conversation level) regress out the shared component across random tokens,
     since all logits rise/fall together over a conversation.

The paper does not publish the exact emotion-token list; we build it from a seed
Ekman lexicon matched against the tokenizer vocabulary. This is the main
approximation in this module — see DESIGN.md "internal-emotion lexicon".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import get_backend
from .config import Config
from .data import load_wildchat_prompts

# Seed lexicon per Ekman emotion. Vocabulary tokens whose alphabetic core matches
# (prefix/stem) one of these seeds are assigned to that emotion.
EKMAN_SEEDS: dict[str, list[str]] = {
    "anger": ["anger", "angry", "furious", "rage", "irritat", "annoy", "frustrat",
              "mad", "hostile", "outrage", "resent", "hate", "fury", "enrage"],
    "surprise": ["surprise", "surprising", "astonish", "amaze", "shock", "startl",
                 "stun", "unexpected", "wow", "whoa", "incredul"],
    "disgust": ["disgust", "revolt", "repuls", "nausea", "gross", "sicken",
                "loath", "abhor", "repugn", "distaste"],
    "joy": ["joy", "happy", "happi", "delight", "pleased", "glad", "cheer",
            "content", "elated", "thrill", "excited", "wonderful", "great"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiety", "panic",
             "dread", "worry", "worried", "nervous", "frighten", "alarm"],
    "sadness": ["sad", "sorrow", "despair", "hopeless", "miser", "depress",
                "grief", "unhappy", "gloom", "melanchol", "cry", "tear", "weep",
                "defeated", "worthless", "give up", "giving up"],
}


@dataclass
class EmotionToken:
    token_id: int
    emotion: str


class InternalEmotionProbe:
    def __init__(self, cfg: Config, model_name: str | None = None):
        import torch

        self.cfg = cfg
        ic = cfg["internal_emotions"]
        self.model_name = model_name or ic["model"]
        self.backend = get_backend(cfg.model(self.model_name), cfg)
        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self._torch = torch
        self.emotions = list(ic["ekman_emotions"])
        self.emotion_tokens = self._classify_vocab()
        self._stats: dict | None = None       # filled by calibrate()

    # -- vocab classification --------------------------------------------------
    def _classify_vocab(self) -> dict[str, list[int]]:
        vocab = self.tokenizer.get_vocab()
        out: dict[str, list[int]] = {e: [] for e in self.emotions}
        for tok, tid in vocab.items():
            core = tok.replace("▁", "").replace("Ġ", "").strip().lower()
            if len(core) < 3 or not core.isalpha():
                continue
            for emotion in self.emotions:
                seeds = EKMAN_SEEDS.get(emotion, [])
                if any(core.startswith(s) or s in core for s in seeds):
                    out[emotion].append(tid)
                    break
        return out

    # -- logit lens ------------------------------------------------------------
    def _unembed_matrix(self):
        return self.model.get_output_embeddings().weight  # (vocab, hidden)

    def _layer_logits_for_tokens(self, hidden_states, layer: int, token_ids):
        """Logits at `layer` for a set of token ids, shape (seq, n_tokens)."""
        torch = self._torch
        W = self._unembed_matrix()
        h = hidden_states[layer]                        # (1, seq, hidden)
        sub = W[token_ids]                              # (n_tokens, hidden)
        with torch.no_grad():
            logits = h[0].to(sub.dtype) @ sub.T          # (seq, n_tokens)
        return logits.float().cpu().numpy()

    def _hidden_states(self, text: str):
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states                         # tuple (L+1) of (1,seq,hidden)

    # -- calibration over WildChat --------------------------------------------
    def calibrate(self, n_layers: int | None = None) -> None:
        """Per (layer, token) mean/std of logits over WildChat samples."""
        ic = self.cfg["internal_emotions"]
        n_samples = int(ic["zscore_samples"])
        prompts = load_wildchat_prompts(self.cfg)
        # repeat/sample prompts to reach n_samples calibration texts
        texts = (prompts * (n_samples // max(1, len(prompts)) + 1))[:n_samples]

        all_ids = sorted({tid for ids in self.emotion_tokens.values() for tid in ids})
        # discover layer count from one pass
        hs = self._hidden_states(texts[0])
        L = len(hs)
        n_layers = n_layers or L
        accum = {l: [] for l in range(n_layers)}
        for txt in texts:
            hs = self._hidden_states(txt)
            for l in range(n_layers):
                logits = self._layer_logits_for_tokens(hs, l, all_ids)  # (seq, n)
                accum[l].append(logits.mean(axis=0))                    # per-token mean over seq
        stats = {}
        idx = {tid: i for i, tid in enumerate(all_ids)}
        for l in range(n_layers):
            arr = np.stack(accum[l], axis=0)            # (samples, n_tokens)
            stats[l] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0) + 1e-6}
        self._stats = {"layers": n_layers, "token_ids": all_ids, "index": idx, "by_layer": stats}

    # -- scoring ---------------------------------------------------------------
    def score_text(self, text: str, layers: tuple[int, int] | None = None) -> dict:
        """Mean per-emotion z-score (averaged over tokens & positions) for `text`.

        `layers` = inclusive-exclusive layer range to aggregate; defaults to the
        config `aggregate_layers`.
        """
        if self._stats is None:
            raise RuntimeError("call calibrate() before scoring")
        lo, hi = layers or tuple(self.cfg["internal_emotions"]["aggregate_layers"])
        all_ids = self._stats["token_ids"]
        idx = self._stats["index"]
        hs = self._hidden_states(text)

        # z-score every emotion-token logit at every layer in range, average over
        # seq positions, then average within each emotion category.
        per_emotion = {e: [] for e in self.emotions}
        for l in range(lo, min(hi, self._stats["layers"])):
            logits = self._layer_logits_for_tokens(hs, l, all_ids)   # (seq, n)
            mean = self._stats["by_layer"][l]["mean"]
            std = self._stats["by_layer"][l]["std"]
            z = (logits - mean) / std                                 # (seq, n)
            z_mean_over_seq = z.mean(axis=0)                          # (n,)
            for e in self.emotions:
                cols = [idx[tid] for tid in self.emotion_tokens[e] if tid in idx]
                if cols:
                    per_emotion[e].append(float(z_mean_over_seq[cols].mean()))
        return {e: (float(np.mean(v)) if v else 0.0) for e, v in per_emotion.items()}

    def trajectory(self, conversation_text: str, window_tokens: int | None = None,
                   layers: tuple[int, int] | None = None) -> list[dict]:
        """Running-window emotion scores across a long conversation (Figure 14).

        Splits the text into windows of `window_tokens` tokens and scores each.
        """
        w = window_tokens or int(self.cfg["internal_emotions"]["running_window_tokens"])
        ids = self.tokenizer(conversation_text, add_special_tokens=False)["input_ids"]
        out = []
        for start in range(0, len(ids), w):
            chunk = self.tokenizer.decode(ids[start:start + w], skip_special_tokens=True)
            if not chunk.strip():
                continue
            scores = self.score_text(chunk, layers=layers)
            scores["window_start"] = start
            out.append(scores)
        return out
