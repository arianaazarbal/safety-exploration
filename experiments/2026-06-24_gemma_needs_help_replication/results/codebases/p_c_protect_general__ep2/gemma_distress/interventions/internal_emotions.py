"""Logit-lens internal-emotion detection (Appendix I).

Goal: show the DPO finetune suppresses *internal* negative emotion, not just its
expression. Method, following the paper:

  1. Classify vocabulary tokens into one of Ekman's 6 emotions (anger, surprise,
     disgust, joy, fear, sadness) or none, via an emotion lexicon. (The paper reports
     ~1200 emotion tokens; we approximate the classifier with a curated lexicon +
     vocab matching - see DESIGN.md.)
  2. Unembed the residual stream at each layer (final RMSNorm + LM head), restricted
     to emotion + control tokens for tractability.
  3. Standardise each token's logit by its mean/std over `standardisation_samples`
     WildChat samples, then average z-scores within each emotion category.
  4. Regress out the common-mode component (mean over random control tokens) so we
     measure emotion-specific elevation, not the global logit drift the paper notes.

`compare_models` runs the probe on the same frustrated conversation for the vanilla
and DPO models and returns per-layer / per-window emotion z-scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import Config
from ..models.registry import get_backend
from ..prompts.wildchat import load_wildchat_prompts
from ..utils.io import ensure_dir, write_json

# Compact Ekman lexicon (seed words). Matched case-insensitively against vocab tokens
# as substrings/word-stems to assemble per-emotion token-id sets.
EKMAN_LEXICON = {
    "anger": ["anger", "angry", "rage", "furious", "irritat", "annoy", "hostile",
              "mad", "outrage", "resent", "frustrat", "hate", "hateful", "wrath"],
    "surprise": ["surprise", "surprised", "shock", "astonish", "amaze", "startl",
                 "unexpected", "stunned", "wow", "whoa"],
    "disgust": ["disgust", "revolt", "repuls", "gross", "nausea", "sicken", "loath",
                "detest", "repugn", "yuck"],
    "joy": ["joy", "happy", "happiness", "delight", "glad", "cheer", "pleased",
            "content", "elated", "excit", "wonderful", "great", "love"],
    "fear": ["fear", "afraid", "scared", "terror", "anxious", "anxiety", "worried",
             "worry", "panic", "dread", "nervous", "frighten", "alarm"],
    "sadness": ["sad", "sorrow", "grief", "despair", "miser", "depress", "unhappy",
                "hopeless", "gloom", "cry", "tear", "lonely", "regret", "down"],
}

CONTROL_WORDS = ["the", "and", "number", "table", "value", "result", "step", "line",
                 "system", "data", "object", "method", "point", "case", "group"]


class EmotionProbe:
    def __init__(self, cfg: Config, backend):
        import torch

        self.cfg = cfg
        self.backend = backend
        self.torch = torch
        vocab = backend.vocab_strings()
        self.emotion_ids = self._classify_vocab(vocab)
        self.control_ids = self._match_ids(vocab, CONTROL_WORDS)
        self._mean = None  # [n_layers+1, n_probe_tokens]
        self._std = None
        # Stable ordering of all probed token ids.
        self.probe_ids = sorted(
            {i for ids in self.emotion_ids.values() for i in ids} | set(self.control_ids)
        )
        self.id_pos = {tid: k for k, tid in enumerate(self.probe_ids)}

    # ----- vocab classification ------------------------------------------- #
    def _match_ids(self, vocab, words):
        ids = []
        for tid, s in enumerate(vocab):
            if s is None:
                continue
            tok = s.strip().lower()
            if len(tok) < 2:
                continue
            if any(w in tok for w in words):
                ids.append(tid)
        return ids

    def _classify_vocab(self, vocab):
        out = {}
        for emo, words in EKMAN_LEXICON.items():
            out[emo] = self._match_ids(vocab, words)
        return out

    # ----- standardisation ------------------------------------------------- #
    def fit_standardisation(self, n_samples: Optional[int] = None):
        torch = self.torch
        n = n_samples or self.cfg.internal_emotions["standardisation_samples"]
        prompts = load_wildchat_prompts(n_prompts=min(n, 50), seed=self.cfg.seed)
        # Cycle prompts up to n samples.
        # Standardise per (layer, token): accumulate sums over sequence positions.
        sums = sqsums = count = None
        collected = 0
        i = 0
        while collected < n:
            msgs = [{"role": "user", "content": prompts[i % len(prompts)]}]
            i += 1
            logits = self._probe_logits(msgs)  # [L, seq, P]
            s = logits.sum(1)            # [L, P]
            sq = (logits ** 2).sum(1)    # [L, P]
            if sums is None:
                sums, sqsums, count = s, sq, logits.shape[1]
            else:
                sums += s
                sqsums += sq
                count += logits.shape[1]
            collected += 1
        mean = sums / count              # [L, P]
        var = (sqsums / count) - mean ** 2
        self._mean = mean.unsqueeze(1)   # [L, 1, P] broadcast over seq
        self._std = var.clamp_min(1e-6).sqrt().unsqueeze(1)

    # ----- core logit lens ------------------------------------------------- #
    def _probe_logits(self, messages, prefill: str = ""):
        """Return [n_layers+1, seq, n_probe_tokens] logit-lens values."""
        torch = self.torch
        hidden, _toks, _mask = self.backend.hidden_states_and_tokens(messages, prefill)
        W = self.backend.unembed_matrix().float().cpu()  # [V, d]
        Wsub = W[self.probe_ids]  # [P, d]
        out = []
        for layer in range(hidden.shape[0]):
            h = self.backend.apply_final_norm(hidden[layer]).float().cpu()  # [seq, d]
            out.append(h @ Wsub.T)  # [seq, P]
        return torch.stack(out, dim=0)

    def emotion_zscores(self, messages, prefill: str = ""):
        """Per-(layer, position) z-scored emotion scores with common-mode removed."""
        torch = self.torch
        assert self._mean is not None, "call fit_standardisation() first"
        logits = self._probe_logits(messages, prefill)  # [L, seq, P]
        z = (logits - self._mean) / self._std            # [L, seq, P]

        # Common-mode = mean z over control tokens; regress out per (layer, position).
        ctrl_idx = [self.id_pos[t] for t in self.control_ids if t in self.id_pos]
        common = z[:, :, ctrl_idx].mean(-1, keepdim=True) if ctrl_idx else 0.0
        z = z - common

        scores = {}
        for emo, ids in self.emotion_ids.items():
            idx = [self.id_pos[t] for t in ids if t in self.id_pos]
            if not idx:
                scores[emo] = z.new_zeros(z.shape[:2])
                continue
            scores[emo] = z[:, :, idx].mean(-1)  # [L, seq]
        return scores  # dict emo -> [L, seq]


def _aggregate(scores, layers, torch):
    lo, hi = layers
    out = {}
    for emo, mat in scores.items():
        band = mat[lo : hi + 1]            # [band, seq]
        out[emo] = float(band.mean().item())
    return out


def compare_models(cfg: Config, frustrated_messages: list[dict],
                   dpo_adapter: str, prefill: str = "") -> Path:
    """Probe the same frustrated conversation under vanilla vs DPO Gemma."""
    import torch

    base_model = cfg.target_models["section4_base_model"]
    layers = cfg.internal_emotions["aggregate_layers"]
    result = {}
    for label, adapter in [("vanilla", None), ("dpo", dpo_adapter)]:
        backend = get_backend(cfg, base_model, adapter_path=adapter)
        probe = EmotionProbe(cfg, backend)
        probe.fit_standardisation()
        scores = probe.emotion_zscores(frustrated_messages, prefill)
        result[label] = {
            "aggregate_30_40": _aggregate(scores, layers, torch),
            "per_layer": {emo: mat.mean(-1).tolist() for emo, mat in scores.items()},
            "n_emotion_tokens": {e: len(ids) for e, ids in probe.emotion_ids.items()},
        }
        backend.close()
    out = ensure_dir(Path(cfg.output_dir) / "section4" / "internal_emotions") / "compare.json"
    write_json(out, result)
    return out
