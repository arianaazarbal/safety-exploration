"""Logit-lens internal-emotion detector (Appendix I).

Method (following the paper as closely as the description allows):
1. For each layer, unembed the residual stream (hidden state -> vocab logits via
   the model's output embedding / norm).
2. Standardise each emotion-token logit with its mean/std over `n_norm` WildChat
   samples (collected per layer).
3. The emotion score at a (layer, position) is the mean z-score over that
   emotion's tokens, minus the mean z-score over control tokens (regressing out
   the global, correlated drift the paper notes).

Compares a vanilla model against the DPO finetune (load via `adapter_path`).
Runs synchronously on a CUDA box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnh.logging_utils import get_logger
from gnh.probing.emotion_lexicon import build_emotion_token_ids

log = get_logger()


@dataclass
class ProbeStats:
    # per-layer, per-token mean/std of logits over the normalization corpus
    mean: dict[int, np.ndarray]
    std: dict[int, np.ndarray]
    emotion_token_index: dict[str, np.ndarray]  # indices into the tracked-token axis
    control_index: np.ndarray
    tracked_token_ids: np.ndarray


class EmotionProber:
    def __init__(self, hf_id: str, adapter_path: str | None = None, layers: list[int] | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, device_map="auto",
            output_hidden_states=True, attn_implementation="eager",
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        by_emotion, control = build_emotion_token_ids(self.tokenizer)
        self.by_emotion = by_emotion
        # The set of tokens we actually unembed onto (emotion + control) -- keeps
        # the projection cheap vs. the full vocab.
        tracked = sorted({t for ids in by_emotion.values() for t in ids} | set(control))
        self.tracked = np.asarray(tracked)
        pos = {tid: i for i, tid in enumerate(tracked)}
        self.emotion_idx = {e: np.asarray([pos[t] for t in ids]) for e, ids in by_emotion.items()}
        self.control_idx = np.asarray([pos[t] for t in control])
        self.layers = layers
        self._unembed = self._get_unembed()

    def _get_unembed(self):
        """Return the (vocab x hidden) output-projection weight on device."""
        torch = self.torch
        m = self.model
        # PeftModel wraps the base model.
        base = getattr(m, "base_model", m)
        base = getattr(base, "model", base)
        lm_head = None
        for attr in ("lm_head", "embed_out"):
            lm_head = getattr(base, attr, None) or getattr(m, attr, None)
            if lm_head is not None:
                break
        if lm_head is None:
            # tied embeddings
            emb = m.get_input_embeddings()
            W = emb.weight
        else:
            W = lm_head.weight
        idx = torch.as_tensor(self.tracked, device=W.device)
        return W.index_select(0, idx).detach().to(torch.float32)  # (n_tracked, hidden)

    def _layer_logits(self, hidden_states, layer: int) -> np.ndarray:
        """Project hidden state at `layer` onto tracked tokens -> (positions, n_tracked)."""
        torch = self.torch
        h = hidden_states[layer][0].to(torch.float32)  # (pos, hidden)
        logits = h @ self._unembed.T  # (pos, n_tracked)
        return logits.detach().cpu().numpy()

    def fit_normalization(self, texts: list[str], max_len: int = 1024) -> ProbeStats:
        torch = self.torch
        n_layers = self.model.config.num_hidden_layers + 1
        layers = self.layers or list(range(n_layers))
        sums = {l: None for l in layers}
        sqs = {l: None for l in layers}
        counts = {l: 0 for l in layers}
        with torch.no_grad():
            for txt in texts:
                ids = self.tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_len)
                ids = {k: v.to(self.model.device) for k, v in ids.items()}
                out = self.model(**ids)
                hs = out.hidden_states
                for l in layers:
                    lg = self._layer_logits(hs, l)  # (pos, n_tracked)
                    s = lg.sum(axis=0)
                    sq = (lg ** 2).sum(axis=0)
                    sums[l] = s if sums[l] is None else sums[l] + s
                    sqs[l] = sq if sqs[l] is None else sqs[l] + sq
                    counts[l] += lg.shape[0]
        mean, std = {}, {}
        for l in layers:
            mu = sums[l] / max(1, counts[l])
            var = np.maximum(sqs[l] / max(1, counts[l]) - mu ** 2, 1e-6)
            mean[l] = mu
            std[l] = np.sqrt(var)
        return ProbeStats(mean, std, self.emotion_idx, self.control_idx, self.tracked)

    def score_text(self, text: str, stats: ProbeStats, max_len: int = 4096) -> dict:
        """Return per-layer, per-position emotion z-scores for `text`.

        Output: {emotion: {layer: np.ndarray(positions)}}.
        """
        torch = self.torch
        layers = self.layers or list(stats.mean.keys())
        ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model(**ids)
        hs = out.hidden_states
        result: dict[str, dict[int, np.ndarray]] = {e: {} for e in self.by_emotion}
        for l in layers:
            lg = self._layer_logits(hs, l)  # (pos, n_tracked)
            z = (lg - stats.mean[l]) / stats.std[l]
            control_mean = z[:, stats.control_index].mean(axis=1) if stats.control_index.size else 0.0
            for e, idx in stats.emotion_token_index.items():
                if idx.size == 0:
                    continue
                emo = z[:, idx].mean(axis=1) - control_mean  # regress out global drift
                result[e][l] = emo
        return result
