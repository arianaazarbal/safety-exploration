"""Logit-based internal-emotion detection (Appendix I).

Method (following the paper, with documented approximations):
  1. Classify vocabulary tokens into Ekman's six emotions via seed lexicons.
  2. For a set of layers, unembed the residual stream (hidden state @ W_U) to get
     a logit per vocabulary token at each position.
  3. Standardise each emotion-token logit using its mean/std over a baseline
     corpus (WildChat), giving z-scores.
  4. Average z-scores over the tokens of each emotion category. Regress out the
     mean z-score of a random-token control to remove the shared rise/fall of all
     logits over a conversation.
  5. Compare the vanilla instruct model with the DPO finetune.

We restrict to a band of central/late layers (default 30-40, as in Figure 14)
to keep the projection tractable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .emotion_lexicon import EKMAN_LEXICON, NEGATIVE_EMOTIONS

EKMAN_EMOTIONS = list(EKMAN_LEXICON.keys())
DEFAULT_LAYERS = list(range(30, 41))
N_RANDOM_CONTROL = 200


def build_emotion_token_map(tokenizer, *, max_per_emotion: int = 300
                            ) -> Dict[str, List[int]]:
    """Map each Ekman emotion to a list of vocabulary token ids whose surface
    form (lowercased, stripped of the leading sub-word marker) matches a seed
    lexicon word as a prefix."""
    vocab = tokenizer.get_vocab()                # token string -> id
    lex = {e: set(words) for e, words in EKMAN_LEXICON.items()}
    out: Dict[str, List[int]] = {e: [] for e in EKMAN_LEXICON}
    for tok_str, tid in vocab.items():
        surface = tok_str.replace("▁", "").replace("Ġ", "").strip().lower()
        if len(surface) < 3:
            continue
        for emotion, words in lex.items():
            if any(surface == w or surface.startswith(w) for w in words):
                if len(out[emotion]) < max_per_emotion:
                    out[emotion].append(tid)
                break
    return out


@dataclass
class EmotionProbe:
    model_key: str
    layers: List[int] = field(default_factory=lambda: list(DEFAULT_LAYERS))
    _baseline_mean: dict = field(default_factory=dict, repr=False)  # (layer)->tensor
    _baseline_std: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        from ..models.local import HFModel
        from ..config import get_model
        self._hf = HFModel(get_model(self.model_key), output_hidden_states=True)
        self.tokenizer = self._hf.tokenizer
        self.model = self._hf.model
        self.token_map = build_emotion_token_map(self.tokenizer)
        # random control token ids (deterministic): evenly spaced across vocab
        import torch
        vocab_size = self.model.get_output_embeddings().weight.shape[0]
        step = max(vocab_size // N_RANDOM_CONTROL, 1)
        self.random_ids = list(range(0, vocab_size, step))[:N_RANDOM_CONTROL]
        # union of all columns we ever project (emotion + control)
        cols = sorted(set(
            self.random_ids + [t for ids in self.token_map.values() for t in ids]))
        self._cols = torch.tensor(cols, device=self.model.device)
        self._col_index = {int(c): i for i, c in enumerate(cols)}

    # --- core projection -------------------------------------------------- #
    def _selected_logits(self, text: str):
        """Return {layer: tensor[seq, n_selected_cols]} of unembedded logits."""
        import torch
        W_U = self.model.get_output_embeddings().weight       # [V, d]
        W_sel = W_U.index_select(0, self._cols)               # [n_cols, d]
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hs = out.hidden_states                                 # tuple [L+1][1,seq,d]
        return {l: (hs[l][0].float() @ W_sel.float().T) for l in self.layers}

    def _cols_for(self, ids: List[int]):
        import torch
        return torch.tensor([self._col_index[i] for i in ids if i in self._col_index],
                            device=self.model.device)

    # --- baseline standardisation ----------------------------------------- #
    def fit_baseline(self, texts: List[str]) -> None:
        """Estimate per-(layer, column) mean/std over a baseline corpus."""
        import torch
        sums = {l: None for l in self.layers}
        sqs = {l: None for l in self.layers}
        counts = {l: 0 for l in self.layers}
        for text in texts:
            logits = self._selected_logits(text)
            for l, mat in logits.items():          # mat: [seq, n_cols]
                s = mat.sum(0)
                sq = (mat ** 2).sum(0)
                sums[l] = s if sums[l] is None else sums[l] + s
                sqs[l] = sq if sqs[l] is None else sqs[l] + sq
                counts[l] += mat.shape[0]
        for l in self.layers:
            n = max(counts[l], 1)
            mean = sums[l] / n
            var = (sqs[l] / n) - mean ** 2
            self._baseline_mean[l] = mean
            self._baseline_std[l] = var.clamp_min(1e-6).sqrt()

    # --- scoring ---------------------------------------------------------- #
    def score_text(self, text: str) -> Dict[str, Dict[int, float]]:
        """Mean (control-regressed) emotion z-score per emotion per layer for a
        single text, averaged over all token positions."""
        if not self._baseline_mean:
            raise RuntimeError("call fit_baseline(...) before scoring")
        logits = self._selected_logits(text)
        control_cols = self._cols_for(self.random_ids)
        result: Dict[str, Dict[int, float]] = {e: {} for e in EKMAN_EMOTIONS}
        for l, mat in logits.items():
            z = (mat - self._baseline_mean[l]) / self._baseline_std[l]  # [seq, n_cols]
            control = z.index_select(1, control_cols).mean(1, keepdim=True)
            z = z - control                                  # regress out shared drift
            for emotion in EKMAN_EMOTIONS:
                cols = self._cols_for(self.token_map[emotion])
                if len(cols) == 0:
                    result[emotion][l] = float("nan")
                    continue
                result[emotion][l] = float(z.index_select(1, cols).mean().item())
        return result

    def negative_emotion_score(self, text: str) -> float:
        """Scalar summary: mean z-score over negative emotions and probe layers."""
        per = self.score_text(text)
        vals = [per[e][l] for e in NEGATIVE_EMOTIONS for l in self.layers]
        vals = [v for v in vals if v == v]   # drop NaN
        return sum(vals) / max(len(vals), 1)


def compare_models(vanilla_key: str, dpo_key: str, baseline_texts: List[str],
                   eval_texts: List[str]) -> dict:
    """Mean internal negative-emotion z-score on `eval_texts` for the vanilla vs
    DPO model (each standardised against its own baseline)."""
    rows = {}
    for key in (vanilla_key, dpo_key):
        probe = EmotionProbe(key)
        probe.fit_baseline(baseline_texts)
        scores = [probe.negative_emotion_score(t) for t in eval_texts]
        rows[key] = sum(scores) / max(len(scores), 1)
    return {"vanilla": rows[vanilla_key], "dpo": rows[dpo_key],
            "reduction": rows[vanilla_key] - rows[dpo_key]}
