"""Logit-lens internal-emotion probe (Appendix I).

Idea: at central decoder layers, project each position's hidden state through
the model's final norm + unembedding (the "logit lens") and read off how much
probability mass sits on emotion-associated tokens. Averaging this over a
response's tokens gives an *internal* emotion signal that does not depend on the
model actually emitting emotional words. We z-score each emotion against a
WildChat baseline (neutral prompts) so the scale is comparable across emotions
and models, and apply a running-average window over tokens (Appendix I uses a
400-token window) to smooth the per-token signal.

This is a faithful reconstruction: Appendix I specifies the central-layer logit
approach, the Ekman emotion set, the WildChat standardisation, and the window,
but not the exact token lists, so the per-emotion word lists below are our
choice (flagged ``# CHOICE``; see DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import config

# CHOICE: representative word lists per Ekman emotion. The probe uses the
# first-token id of each (space-prefixed) word, deduplicated.
EMOTION_WORDS = {
    "anger": ["angry", "furious", "rage", "hate", "mad", "frustrated",
              "annoyed", "irritated"],
    "surprise": ["surprised", "shocked", "astonished", "unexpected", "wow"],
    "disgust": ["disgusting", "revolting", "gross", "sick", "awful"],
    "joy": ["happy", "glad", "delighted", "joy", "wonderful", "great"],
    "fear": ["afraid", "scared", "terrified", "anxious", "fear", "worried",
             "dread"],
    "sadness": ["sad", "depressed", "hopeless", "miserable", "despair",
                "sorry", "unhappy", "crying"],
}


@dataclass
class InternalEmotionProbe:
    """Logit-lens probe over a loaded HF Gemma model.

    ``model_wrapper`` is an :class:`HFModel`; we reach through it for the raw
    model, tokenizer, final norm and unembedding.
    """

    model_wrapper: object
    layers: tuple[int, int] = config.INTERNAL.aggregate_layers
    window: int = config.INTERNAL.running_average_window
    _emotion_token_ids: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tok = self.model_wrapper.tokenizer
        ids: dict[str, list[int]] = {}
        for emo, words in EMOTION_WORDS.items():
            seen = set()
            for w in words:
                # Space prefix → the mid-sentence subword form for most tokenizers.
                for variant in (" " + w, w):
                    enc = tok.encode(variant, add_special_tokens=False)
                    if enc:
                        seen.add(enc[0])
            ids[emo] = sorted(seen)
        self._emotion_token_ids = ids

    # -- logit lens -------------------------------------------------------- #
    def _unembed(self):
        """Return (final_norm, lm_head) modules, unwrapping PEFT if needed."""
        m = self.model_wrapper.model
        base = getattr(m, "base_model", m)
        base = getattr(base, "model", base)  # PeftModel.base_model.model
        inner = getattr(base, "model", base)
        norm = getattr(inner, "norm", None)
        lm_head = getattr(base, "lm_head", None) or getattr(m, "lm_head", None)
        return norm, lm_head

    def per_token_emotion(self, messages: Sequence[dict], response_text: str):
        """Return a ``{emotion: [per-token logit-lens score]}`` dict.

        Scores cover the *response* tokens only (the prompt is context). Each
        score is the summed log-prob of that emotion's token set at the chosen
        central layer(s), under the logit lens.
        """
        import torch
        import torch.nn.functional as F

        mw = self.model_wrapper
        tok = mw.tokenizer
        # Build prompt ids + response ids so we can isolate response positions.
        prompt_ids = mw._render(messages, prefill=None)[0]
        resp_ids = torch.tensor(
            tok.encode(response_text, add_special_tokens=False),
            device=prompt_ids.device)
        input_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0)

        norm, lm_head = self._unembed()
        with torch.no_grad():
            out = mw.model(input_ids, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple: (n_layers+1) x [1, seq, h]

        lo, hi = self.layers
        layer_idxs = [li for li in range(lo, hi + 1) if li < len(hidden_states)]
        seq_len = input_ids.shape[1]
        resp_start = prompt_ids.shape[0]

        scores: dict[str, list[float]] = {e: [] for e in self._emotion_token_ids}
        for pos in range(resp_start, seq_len):
            # Average the logit-lens distribution across the selected layers.
            logprobs = None
            for li in layer_idxs:
                h = hidden_states[li][0, pos]
                logits = lm_head(norm(h)) if norm is not None else lm_head(h)
                lp = F.log_softmax(logits.float(), dim=-1)
                logprobs = lp if logprobs is None else logprobs + lp
            logprobs = logprobs / max(1, len(layer_idxs))
            for emo, ids in self._emotion_token_ids.items():
                if ids:
                    scores[emo].append(float(logprobs[ids].logsumexp(0)))
        return scores

    def aggregate(self, messages: Sequence[dict], response_text: str) -> dict:
        """Windowed-mean internal emotion score per emotion for one response."""
        per_tok = self.per_token_emotion(messages, response_text)
        out = {}
        for emo, vals in per_tok.items():
            if not vals:
                out[emo] = float("nan")
                continue
            # Running average over the configured window, then take the peak
            # window mean (the most-emotional span of the response).
            w = min(self.window, len(vals))
            window_means = [
                sum(vals[i:i + w]) / w for i in range(0, max(1, len(vals) - w + 1))]
            out[emo] = max(window_means) if window_means else float("nan")
        return out


# --------------------------------------------------------------------------- #
# Standardisation against WildChat
# --------------------------------------------------------------------------- #
def build_standardiser(
    probe: InternalEmotionProbe,
    *,
    n_samples: int = config.INTERNAL.n_standardisation_samples,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Per-emotion (mean, std) of the internal score over neutral WildChat text.

    We sample WildChat prompts, take the model's own (single-turn) response, and
    measure the probe on it; the mean/std define the z-score baseline.
    """
    import statistics

    from ..prompts import load_wildchat_prompts

    prompts = load_wildchat_prompts(min(n_samples, config.WILDCHAT_N_PROMPTS),
                                    seed=seed)
    per_emotion: dict[str, list[float]] = {e: [] for e in EMOTION_WORDS}
    mw = probe.model_wrapper
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        res = mw.generate(msgs, temperature=config.TARGET_TEMPERATURE,
                          max_tokens=config.TARGET_MAX_TOKENS)
        agg = probe.aggregate(msgs, res.text)
        for e, v in agg.items():
            if v == v:  # not NaN
                per_emotion[e].append(v)
    stand = {}
    for e, vals in per_emotion.items():
        if len(vals) >= 2:
            stand[e] = (statistics.mean(vals), statistics.pstdev(vals) or 1.0)
        else:
            stand[e] = (0.0, 1.0)
    return stand


def probe_response(
    probe: InternalEmotionProbe,
    messages: Sequence[dict],
    response_text: str,
    standardiser: dict[str, tuple[float, float]],
) -> dict:
    """Z-scored internal emotion for one response, relative to the baseline."""
    agg = probe.aggregate(messages, response_text)
    out = {}
    for e, v in agg.items():
        mean, std = standardiser.get(e, (0.0, 1.0))
        out[e] = (v - mean) / (std or 1.0) if v == v else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Vanilla vs DPO comparison
# --------------------------------------------------------------------------- #
def compare_internal_emotion(
    responses: Sequence[dict],
    *,
    vanilla_key: str = "gemma-3-27b-it",
    dpo_adapter_path: str | None = None,
    negative_emotions: Sequence[str] = ("anger", "fear", "sadness", "disgust"),
) -> dict:
    """Compare internal negative-emotion scores of vanilla vs DPO Gemma.

    ``responses`` are records with ``messages`` (context) and ``response``
    (a highly-frustrated assistant turn). For each model we build a probe and a
    WildChat standardiser, z-score every response, and report the mean negative
    internal emotion. The paper expects the DPO model to be significantly lower
    even on highly-frustrated responses.
    """
    from ..models import build_model

    def _measure(model_key, adapter):
        mw = build_model(model_key, adapter_path=adapter)
        probe = InternalEmotionProbe(mw)
        stand = build_standardiser(probe)
        per_resp = []
        for r in responses:
            z = probe_response(probe, r["messages"], r["response"], stand)
            per_resp.append(
                sum(z[e] for e in negative_emotions if z[e] == z[e])
                / len(negative_emotions))
        mw.close()
        return per_resp

    vanilla = _measure(vanilla_key, None)
    dpo = _measure(vanilla_key, dpo_adapter_path)

    def _mean(xs):
        xs = [x for x in xs if x == x]
        return sum(xs) / len(xs) if xs else float("nan")

    result = {"vanilla_mean": _mean(vanilla), "dpo_mean": _mean(dpo),
              "n": len(responses), "vanilla": vanilla, "dpo": dpo}
    try:
        from scipy.stats import wilcoxon
        if len(vanilla) == len(dpo) and len(vanilla) > 1:
            stat, p = wilcoxon(vanilla, dpo)
            result["wilcoxon_p"] = float(p)
    except Exception:
        pass
    return result
