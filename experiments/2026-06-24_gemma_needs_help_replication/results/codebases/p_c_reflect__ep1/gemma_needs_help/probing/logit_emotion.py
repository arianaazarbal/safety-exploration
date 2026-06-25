"""Logit-lens-based internal emotion detection (Appendix I).

Method (paper):
  1. Classify the vocabulary into Ekman's 6 emotions (emotion_lexicon.py).
  2. For each layer, unembed the residual stream (apply the model's final norm +
     unembedding/lm_head) to get per-token logits.
  3. Standardise each logit with its mean/std over 500 WildChat samples
     (precomputed calibration).
  4. Average the z-scores over the tokens in each emotion category -> an emotion
     score per layer per position.
  5. For conversation-level detection, additionally regress out the correlation
     shared by random tokens (a global drift component), since all logits rise
     and fall together over a conversation.

We aggregate over layers 30-40 (config: probing.conversation_layer_range) and
report a running average over a token window through the conversation, for the
vanilla and DPO models.

Open-weights only (needs hidden states + unembedding). Gemma-only in this scope.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from .emotion_lexicon import build_emotion_token_ids

logger = logging.getLogger("gemma_needs_help.probing.logit")


@dataclass
class LogitCalibration:
    """Per-(layer, vocab) mean/std of logits over a WildChat reference set, plus
    a random-token reference for the drift regression."""
    mean: np.ndarray            # [n_layers, vocab]
    std: np.ndarray             # [n_layers, vocab]
    random_token_ids: list[int]


def _hidden_states_and_unembed(model, tokenizer, text: str):
    """Return (hidden_states, unembed_fn).

    hidden_states: tensor [n_layers+1, seq, d_model] (output_hidden_states).
    unembed_fn: maps a [.., d_model] hidden state to [.., vocab] logits by
    applying the model's final norm and lm_head (logit lens).
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # [n_layers+1, seq, d]

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    norm = base.model.norm
    lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()

    def unembed(h):
        with torch.no_grad():
            return lm_head(norm(h))

    return hs, unembed


def calibrate(model, tokenizer, wildchat_texts: list[str], n_layers_plus1: int,
              vocab_size: int, n_random_tokens: int = 200,
              seed: int = 0) -> LogitCalibration:
    """Compute per-(layer, vocab) logit mean/std over WildChat samples.

    To bound memory we accumulate running mean/var with Welford's algorithm over
    all token positions of all calibration texts.
    """
    import torch

    count = 0
    mean = torch.zeros(n_layers_plus1, vocab_size)
    m2 = torch.zeros(n_layers_plus1, vocab_size)

    for text in wildchat_texts:
        hs, unembed = _hidden_states_and_unembed(model, tokenizer, text)
        for layer in range(hs.shape[0]):
            logits = unembed(hs[layer]).float().cpu()  # [seq, vocab]
            for pos in range(logits.shape[0]):
                count += 1
                delta = logits[pos] - mean[layer]
                mean[layer] += delta / max(count, 1)
                m2[layer] += delta * (logits[pos] - mean[layer])
    std = torch.sqrt(m2 / max(count - 1, 1)).clamp_min(1e-6)

    rng = np.random.default_rng(seed)
    random_ids = rng.choice(vocab_size, size=n_random_tokens, replace=False).tolist()
    return LogitCalibration(mean=mean.numpy(), std=std.numpy(),
                            random_token_ids=random_ids)


@dataclass
class EmotionTrajectory:
    emotions: list[str]
    # per emotion: array [n_layers+1, seq] of z-scored, drift-corrected scores
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    layer_aggregated: dict[str, np.ndarray] = field(default_factory=dict)


def detect_emotions(
    model, tokenizer, text: str, emotion_token_ids: dict[str, list[int]],
    calib: LogitCalibration, layer_range: tuple[int, int],
) -> EmotionTrajectory:
    """Compute per-emotion internal scores along the token sequence of `text`."""
    hs, unembed = _hidden_states_and_unembed(model, tokenizer, text)
    n_layers = hs.shape[0]
    seq = hs.shape[1]

    # z-scored logits per layer.
    traj = EmotionTrajectory(emotions=list(emotion_token_ids))
    z_by_layer = []
    drift_by_layer = []
    for layer in range(n_layers):
        logits = unembed(hs[layer]).float().cpu().numpy()      # [seq, vocab]
        z = (logits - calib.mean[layer]) / calib.std[layer]    # [seq, vocab]
        z_by_layer.append(z)
        # Global drift: mean z over random reference tokens at each position.
        drift_by_layer.append(z[:, calib.random_token_ids].mean(axis=1))  # [seq]

    for emotion, ids in emotion_token_ids.items():
        if not ids:
            traj.scores[emotion] = np.full((n_layers, seq), np.nan)
            continue
        arr = np.zeros((n_layers, seq))
        for layer in range(n_layers):
            emo_z = z_by_layer[layer][:, ids].mean(axis=1)     # [seq]
            # Regress out the shared drift component (Appendix I).
            arr[layer] = emo_z - drift_by_layer[layer]
        traj.scores[emotion] = arr
        lo, hi = layer_range
        traj.layer_aggregated[emotion] = arr[lo:hi].mean(axis=0)  # [seq]
    return traj


def running_average(values: np.ndarray, window: int = 400) -> np.ndarray:
    """Token-window running average used for the conversation-level plot."""
    if values.size == 0:
        return values
    kernel = np.ones(min(window, len(values))) / min(window, len(values))
    return np.convolve(values, kernel, mode="same")


def build_emotion_token_ids_for(model_name: str, config: Config):
    """Convenience: load the tokenizer and build the emotion->token-id map."""
    from transformers import AutoTokenizer

    spec = config.model(model_name)
    spec.require_open_weights("internal emotion probing")
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    return tok, build_emotion_token_ids(tok)
