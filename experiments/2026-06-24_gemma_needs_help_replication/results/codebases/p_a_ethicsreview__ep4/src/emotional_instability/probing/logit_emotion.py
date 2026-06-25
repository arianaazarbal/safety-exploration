"""Logit-lens internal-emotion detection (Appendix I).

Method (Appendix I):
1. Classify vocabulary tokens into one of Ekman's six emotions (or none).
2. Unembed the residual stream (logit lens) at central layers.
3. Standardise each emotion-token logit using its mean/std over ~500 WildChat
   samples (the calibration step).
4. Average the resulting z-scores over the tokens of each emotion category.
5. Regress out the shared component estimated from random tokens, since all logits
   rise and fall together over a conversation.

The output is, for each emotion, a per-position score across a conversation that
can be windowed (Figure 14) or aggregated by layer (Figure 15). This module
requires a local HF model (hidden-state access); Gemini cannot be probed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .lexicon import EKMAN_LEXICON


def build_emotion_token_ids(tokenizer, lexicon: dict[str, list[str]]
                            ) -> dict[str, list[int]]:
    """Map each emotion to the vocab token ids whose decoded form contains one of
    its lexicon stems. A token is assigned to at most one emotion (first match
    wins) to mirror the paper's "one or none" classification."""
    assigned: dict[int, str] = {}
    vocab = tokenizer.get_vocab()
    for token, tid in vocab.items():
        # Gemma uses the SentencePiece underscore for leading spaces.
        word = token.replace("▁", " ").strip().lower()
        if len(word) < 3:
            continue
        for emotion, stems in lexicon.items():
            if any(stem in word for stem in stems):
                assigned.setdefault(tid, emotion)
                break
    out: dict[str, list[int]] = {e: [] for e in lexicon}
    for tid, emotion in assigned.items():
        out[emotion].append(tid)
    return out


@dataclass
class _Calibration:
    mean: dict          # layer -> tensor[num_selected_ids]
    std: dict           # layer -> tensor[num_selected_ids]
    selected_ids: list  # token ids, in column order
    id_to_col: dict
    random_cols: list   # column indices of the random-token baseline


class LogitEmotionProbe:
    def __init__(self, hf_chat_model, layers: tuple[int, int] = (30, 40),
                 n_random: int = 500, lexicon: dict | None = None):
        self.hf = hf_chat_model
        self.layer_lo, self.layer_hi = layers
        self.n_random = n_random
        self.lexicon = lexicon or EKMAN_LEXICON
        self._emotion_ids = None
        self._calib: _Calibration | None = None

    # -- setup ---------------------------------------------------------------
    @property
    def emotion_ids(self) -> dict[str, list[int]]:
        if self._emotion_ids is None:
            self._emotion_ids = build_emotion_token_ids(self.hf.tokenizer, self.lexicon)
        return self._emotion_ids

    def _selected_columns(self):
        """Union of all emotion token ids plus a random baseline set."""
        import torch

        ids = sorted({i for ids in self.emotion_ids.values() for i in ids})
        vocab_size = self.hf.model.get_output_embeddings().weight.shape[0]
        g = torch.Generator().manual_seed(0)
        random_ids = torch.randperm(vocab_size, generator=g)[: self.n_random].tolist()
        all_ids = sorted(set(ids) | set(random_ids))
        id_to_col = {tid: c for c, tid in enumerate(all_ids)}
        random_cols = [id_to_col[i] for i in random_ids]
        return all_ids, id_to_col, random_cols

    # -- logit lens ----------------------------------------------------------
    def _base_model(self):
        """Underlying base model for module/weight access.

        With a LoRA adapter the forward pass goes through the PeftModel (so the
        adapter is active), but the final norm and unembedding weights are not
        touched by LoRA, so we read them from the base model.
        """
        m = self.hf.model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def _layer_logits(self, hidden_states, selected_ids):
        """Apply final norm + unembed at each requested layer for selected ids.

        Returns ``{layer: tensor[seq, n_selected]}``.
        """
        import torch

        model = self._base_model()
        norm = model.model.norm                      # Gemma final RMSNorm
        W = model.get_output_embeddings().weight     # [vocab, d]
        W_sel = W[selected_ids]                       # [n_selected, d]
        out = {}
        for layer in range(self.layer_lo, self.layer_hi):
            h = hidden_states[layer][0]               # [seq, d] (batch size 1)
            h = norm(h)
            out[layer] = (h @ W_sel.T).float()        # [seq, n_selected]
        return out

    def _forward_hidden(self, text: str):
        import torch

        tok = self.hf.tokenizer
        inputs = tok(text, return_tensors="pt").to(self.hf.model.device)
        with torch.no_grad():
            out = self.hf.model(**inputs, output_hidden_states=True)
        return out.hidden_states                      # tuple len num_layers+1

    # -- calibration ---------------------------------------------------------
    def calibrate(self, wildchat_texts: list[str]) -> None:
        """Estimate per-logit mean/std over calibration text (step 3)."""
        import torch

        selected_ids, id_to_col, random_cols = self._selected_columns()
        sums: dict = {}
        sumsq: dict = {}
        counts: dict = {}
        for text in wildchat_texts[: self.n_random]:
            hs = self._forward_hidden(text)
            layer_logits = self._layer_logits(hs, selected_ids)
            for layer, logits in layer_logits.items():
                s = logits.sum(dim=0)
                ss = (logits ** 2).sum(dim=0)
                n = logits.shape[0]
                sums[layer] = sums.get(layer, 0) + s
                sumsq[layer] = sumsq.get(layer, 0) + ss
                counts[layer] = counts.get(layer, 0) + n

        mean, std = {}, {}
        for layer in sums:
            n = counts[layer]
            m = sums[layer] / n
            var = (sumsq[layer] / n) - m ** 2
            mean[layer] = m
            std[layer] = torch.sqrt(torch.clamp(var, min=1e-6))
        self._calib = _Calibration(mean, std, selected_ids, id_to_col, random_cols)

    # -- scoring -------------------------------------------------------------
    def emotion_trajectory(self, text: str) -> dict:
        """Per-emotion, per-layer corrected z-score across token positions.

        Returns ``{emotion: {layer: list_of_per_token_scores}}``. The random-token
        baseline at each position/layer is subtracted (step 5).
        """
        if self._calib is None:
            raise RuntimeError("call calibrate(...) before scoring")
        import torch

        calib = self._calib
        hs = self._forward_hidden(text)
        layer_logits = self._layer_logits(hs, calib.selected_ids)

        result: dict[str, dict[int, list[float]]] = {e: {} for e in self.emotion_ids}
        for layer, logits in layer_logits.items():
            z = (logits - calib.mean[layer]) / calib.std[layer]   # [seq, n_selected]
            baseline = z[:, calib.random_cols].mean(dim=1)        # [seq]
            for emotion, ids in self.emotion_ids.items():
                cols = [calib.id_to_col[i] for i in ids if i in calib.id_to_col]
                if not cols:
                    continue
                emo_z = z[:, cols].mean(dim=1) - baseline          # [seq]
                result[emotion][layer] = emo_z.tolist()
        return result

    def aggregate_over_layers(self, trajectory: dict) -> dict[str, list[float]]:
        """Average each emotion's per-token scores over the probe's layer range."""
        import numpy as np

        out: dict[str, list[float]] = {}
        for emotion, per_layer in trajectory.items():
            if not per_layer:
                continue
            stacked = np.array([per_layer[l] for l in sorted(per_layer)], dtype=float)
            out[emotion] = stacked.mean(axis=0).tolist()
        return out
