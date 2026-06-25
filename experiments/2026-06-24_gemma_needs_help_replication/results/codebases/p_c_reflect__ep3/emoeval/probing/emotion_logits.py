"""Logit-based internal emotion detection (Appendix I.2).

Method (as described in the appendix):
  1. Classify the vocabulary into Ekman's 6 emotions (see ekman.py) -> ~1200
     emotion tokens.
  2. For a given residual stream, unembed (logit lens) and standardise each
     logit with its mean/std computed over 500 WildChat samples.
  3. Average the z-scores over the tokens in an emotion category to get a per-
     emotion score at each layer and each conversation position.
  4. At the conversation level, regress out the correlation between random
     tokens (all logits rise/fall together), leaving an emotion-specific signal.

This module computes (a) baseline per-token logit mean/std over WildChat and
(b) per-emotion z-score trajectories through a conversation, aggregated over a
configurable layer band (the paper uses layers 30-40 for the conversation-level
plot).
"""
from __future__ import annotations

from dataclasses import dataclass

from .ekman import EKMAN_EMOTIONS, build_emotion_token_ids


@dataclass
class BaselineStats:
    mean: object  # tensor [n_layers, vocab]
    std: object   # tensor [n_layers, vocab]


def _layer_logits(hf_client, input_ids):
    """Return logit-lens logits for every layer: tensor [n_layers, seq, vocab].

    Applies the model's final norm + unembedding to each layer's residual
    stream (the standard logit-lens construction)."""
    import torch

    model = hf_client.model
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hidden = out.hidden_states  # tuple length n_layers+1, each [1, seq, hidden]
    # locate final norm + unembedding (works for Gemma-3 decoder layout)
    core = getattr(model, "model", model)
    norm = core.norm
    lm_head = model.get_output_embeddings()
    logits_per_layer = []
    with torch.no_grad():
        for h in hidden[1:]:           # skip embedding layer
            logits_per_layer.append(lm_head(norm(h))[0])  # [seq, vocab]
    return torch.stack(logits_per_layer, dim=0)  # [n_layers, seq, vocab]


def compute_baseline(hf_client, wildchat_texts, *, max_samples: int = 500):
    """Per-token logit mean/std over WildChat samples (Appendix I.2)."""
    import torch

    tok = hf_client.tokenizer
    sums = None
    sqs = None
    count = 0
    for text in wildchat_texts[:max_samples]:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=256)["input_ids"]
        ids = ids.to(hf_client.model.device)
        logits = _layer_logits(hf_client, ids)        # [L, seq, V]
        flat = logits.reshape(logits.shape[0], -1, logits.shape[-1]).mean(dim=1)  # [L, V]
        if sums is None:
            sums = torch.zeros_like(flat)
            sqs = torch.zeros_like(flat)
        sums += flat
        sqs += flat * flat
        count += 1
    mean = sums / max(count, 1)
    var = (sqs / max(count, 1)) - mean * mean
    std = var.clamp_min(1e-6).sqrt()
    return BaselineStats(mean=mean, std=std)


def emotion_trajectory(
    hf_client,
    messages: list[dict],
    baseline: BaselineStats,
    *,
    layer_band: tuple[int, int] = (30, 40),
    regress_random: bool = True,
    n_random_tokens: int = 200,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Per-emotion z-score trajectory over conversation token positions.

    Returns {emotion: [score_per_position]} aggregated over `layer_band`.
    """
    import torch

    tok = hf_client.tokenizer
    emo_ids = build_emotion_token_ids(tok)
    prompt = hf_client._render_prompt(messages, add_generation_prompt=False)
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(hf_client.model.device)

    logits = _layer_logits(hf_client, ids)            # [L, seq, V]
    lo, hi = layer_band
    band = logits[lo:hi]                              # [band, seq, V]
    mean = baseline.mean[lo:hi].unsqueeze(1)          # [band, 1, V]
    std = baseline.std[lo:hi].unsqueeze(1)
    z = (band - mean) / std                           # [band, seq, V] z-scored logits
    z = z.mean(dim=0)                                 # [seq, V] avg over band

    # Optionally regress out the common-mode signal estimated from random tokens
    # (the appendix notes all logits rise/fall together over a conversation).
    if regress_random:
        g = torch.Generator(device=z.device).manual_seed(seed)
        rand_ids = torch.randint(0, z.shape[-1], (n_random_tokens,), generator=g,
                                  device=z.device)
        common = z[:, rand_ids].mean(dim=1, keepdim=True)   # [seq, 1]
        z = z - common

    out: dict[str, list[float]] = {}
    for emotion in EKMAN_EMOTIONS:
        token_ids = emo_ids[emotion]
        if not token_ids:
            out[emotion] = [0.0] * z.shape[0]
            continue
        idx = torch.tensor(token_ids, device=z.device)
        scores = z[:, idx].mean(dim=1)                # [seq]
        out[emotion] = scores.detach().float().cpu().tolist()
    return out
