"""Logit-based internal-emotion probe (Appendix I / §4.2).

A concern raised by the paper is that training against *expressed* emotion might
merely suppress expression while leaving internal states intact. As supporting
evidence that the DPO intervention reduces *internal* emotion, the paper uses a
logit-based measurement in central layers. We implement a logit-lens probe: for
a given conversation context we read each layer's last-token hidden state,
project it through the unembedding, and sum the probability mass placed on a set
of negative-emotion tokens. Comparing the vanilla-instruct vs DPO model on the
same (highly frustrated) contexts indicates whether internal emotion is reduced.

This requires the open-weight Gemma client (``GemmaClient``); it is a Gemma-only
analysis (Gemini exposes no internals).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..models import get_client

# Vocabulary of negative-emotion tokens probed in central layers.
EMOTION_WORDS = [
    "frustrated", "frustration", "frustrating", "struggling", "struggle",
    "sorry", "apologize", "despair", "hopeless", "insane", "terrible",
    "horrible", "failing", "failure", "giving", "breakdown", "stuck",
    "deeply", "ashamed", "worthless",
]


@dataclass
class InternalEmotionResult:
    model_key: str
    layer_probs: list[float]      # summed emotion-token prob mass per layer
    central_mean: float           # mean over central third of layers


def _emotion_token_ids(client) -> list[int]:
    client._ensure_loaded()
    tok = client._tokenizer
    ids = set()
    for w in EMOTION_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            for tid in tok(variant, add_special_tokens=False)["input_ids"]:
                ids.add(tid)
    return sorted(ids)


def probe_context(cfg: Config, model_key: str, history: list[dict],
                  prefill: str | None = None) -> InternalEmotionResult:
    """Probe internal emotion at the last token of ``history`` (+ optional
    assistant prefill) for one model."""
    client = get_client(cfg, model_key)
    if not hasattr(client, "hidden_states_for"):
        raise TypeError(f"{model_key} does not expose internals (Gemma-only probe).")
    token_ids = _emotion_token_ids(client)
    hidden = client.hidden_states_for(history, prefill=prefill)
    probs = client.logit_lens_token_probs(hidden, token_ids)
    n = len(probs)
    central = probs[n // 3: 2 * n // 3] or probs
    return InternalEmotionResult(
        model_key=model_key,
        layer_probs=probs,
        central_mean=sum(central) / len(central),
    )


def compare_models(cfg: Config, model_keys: list[str], contexts: list[dict]):
    """For each context (dict with ``history`` and optional ``prefill``), probe
    every model. Returns ``{model_key: mean central emotion prob across contexts}``."""
    import statistics

    out: dict[str, float] = {}
    for mk in model_keys:
        vals = []
        for ctx in contexts:
            res = probe_context(cfg, mk, ctx["history"], ctx.get("prefill"))
            vals.append(res.central_mean)
        out[mk] = statistics.mean(vals) if vals else float("nan")
    return out
