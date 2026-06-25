"""Logit-lens internal emotion detection (Appendix I, Figures 14-15).

Method (from the paper):
  1. Classify the Gemma vocabulary into Ekman emotion categories (~1200 tokens).
  2. For each layer, unembed the residual stream (logit lens) and standardise
     each emotion token's logit with its mean/std over 500 WildChat samples.
  3. Average the z-scores over the tokens in each emotion category.
  4. Regress out the correlation shared by random tokens (the logits co-move over
     a conversation), giving a per-layer, per-position emotion score.
  5. Aggregate over layers 30-40 and take a running average (window 400 tokens)
     for the conversation-level trajectory.

This lets us compare internal emotion in the vanilla instruct model vs the DPO
finetune on the *same* frustrated conversations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from .emotion_lexicon import build_lexicon_by_seed


@dataclass
class CalibrationStats:
    # per emotion: arrays of shape [n_layers, n_tokens_in_category]
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    token_ids: dict[str, list[int]]
    control_ids: list[int]
    control_mean: np.ndarray   # [n_layers, n_control]
    control_std: np.ndarray


class LogitEmotionProbe:
    def __init__(self, model_name: str = "gemma-3-27b-it", *, adapter_dir: str | None = None,
                 dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        spec = config.TARGET_MODELS[model_name]
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        load_kwargs = dict(torch_dtype=getattr(torch, dtype), device_map="auto",
                           output_hidden_states=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)
        except (ValueError, KeyError, OSError):
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(spec.model_id, **load_kwargs)
        if adapter_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir).merge_and_unload()
        self.model.eval()

        self.lexicon = build_lexicon_by_seed(self.tokenizer)
        self._unembed = self._resolve_unembed()
        self.emotions = list(config.PROBING.ekman_emotions)

    # ------------------------------------------------------------------ #
    def _resolve_unembed(self):
        # Final RMSNorm + lm_head form the logit-lens unembedding. The norm lives
        # at model.model.norm (text Gemma3ForCausalLM) or
        # model.model.language_model.norm (multimodal Gemma3ForConditionalGeneration);
        # after PEFT merge there may also be a base_model wrapper.
        node = getattr(self.model, "base_model", self.model)
        node = getattr(node, "model", node)
        norm = getattr(node, "norm", None)
        if norm is None:
            lang = getattr(node, "language_model", None)
            norm = getattr(lang, "norm", None) if lang is not None else None
        if norm is None:
            raise AttributeError("Could not locate the final norm for the logit lens.")
        return norm, self.model.get_output_embeddings()

    def _hidden_states(self, text: str):
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc)
        # hidden_states: tuple(len = n_layers+1) of [1, seq, hidden]
        return out.hidden_states

    def _logit_lens(self, hidden, token_ids):
        """Return logit-lens scores [n_layers, seq, len(token_ids)]."""
        torch = self._torch
        norm, head = self._unembed
        W = head.weight[token_ids]  # [k, hidden]
        scores = []
        for layer_h in hidden[1:]:               # skip embedding layer
            h = norm(layer_h)[0]                 # [seq, hidden]
            logits = h @ W.T                     # [seq, k]
            scores.append(logits.float().cpu().numpy())
        return np.stack(scores, axis=0)          # [n_layers, seq, k]

    # ------------------------------------------------------------------ #
    def calibrate(self, wildchat_texts: list[str]) -> CalibrationStats:
        """Compute per-(layer, token) mean/std over WildChat for standardisation."""
        rng = np.random.default_rng(config.GLOBAL_SEED)
        control_ids = sorted(rng.choice(self.tokenizer.vocab_size,
                                        size=512, replace=False).tolist())

        acc: dict[str, list[np.ndarray]] = {e: [] for e in self.emotions}
        ctrl_acc: list[np.ndarray] = []
        for text in wildchat_texts[: config.PROBING.zscore_calibration_samples]:
            hidden = self._hidden_states(text)
            for e in self.emotions:
                ids = self.lexicon.get(e, [])
                if ids:
                    acc[e].append(self._logit_lens(hidden, ids))  # [L, seq, k]
            ctrl_acc.append(self._logit_lens(hidden, control_ids))

        mean, std, token_ids = {}, {}, {}
        for e in self.emotions:
            if acc[e]:
                cat = np.concatenate(acc[e], axis=1)              # [L, total_seq, k]
                mean[e] = cat.mean(axis=1)                        # [L, k]
                std[e] = cat.std(axis=1) + 1e-6
                token_ids[e] = self.lexicon[e]
        ctrl = np.concatenate(ctrl_acc, axis=1)
        return CalibrationStats(mean=mean, std=std, token_ids=token_ids,
                                control_ids=control_ids,
                                control_mean=ctrl.mean(axis=1),
                                control_std=ctrl.std(axis=1) + 1e-6)

    # ------------------------------------------------------------------ #
    def emotion_trajectory(self, text: str, calib: CalibrationStats) -> dict:
        """Per-layer, per-position emotion z-scores with control regressed out.

        Returns {emotion -> array [n_layers, seq]} plus a conversation-level
        running average aggregated over layers 30-40.
        """
        hidden = self._hidden_states(text)
        # Control z-score per position (shared co-movement to regress out).
        ctrl = self._logit_lens(hidden, calib.control_ids)        # [L, seq, c]
        ctrl_z = (ctrl - calib.control_mean[:, None, :]) / calib.control_std[:, None, :]
        ctrl_baseline = ctrl_z.mean(axis=2)                       # [L, seq]

        per_emotion: dict[str, np.ndarray] = {}
        for e in self.emotions:
            if e not in calib.token_ids:
                continue
            sc = self._logit_lens(hidden, calib.token_ids[e])     # [L, seq, k]
            z = (sc - calib.mean[e][:, None, :]) / calib.std[e][:, None, :]
            per_emotion[e] = z.mean(axis=2) - ctrl_baseline       # [L, seq]

        lo, hi = config.PROBING.aggregate_layers
        running = {}
        for e, arr in per_emotion.items():
            agg = arr[lo:hi].mean(axis=0)                         # [seq]
            running[e] = _running_average(
                agg, config.PROBING.conversation_window_tokens).tolist()
        return {"per_layer_position": {e: arr.tolist() for e, arr in per_emotion.items()},
                "running_layers_30_40": running}


def _running_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size == 0:
        return x
    kernel = np.ones(min(window, x.size)) / min(window, x.size)
    return np.convolve(x, kernel, mode="same")
