"""Logit-based internal-emotion detection (Appendix I).

Method (Appendix I): classify every token in the Gemma vocabulary as describing
one of Ekman's six basic emotions (anger, surprise, disgust, joy, fear,
sadness) or none. To score an emotion at a given point in a conversation, we
unembed the residual stream, standardise each vocab logit by its mean/std over
500 WildChat samples (z-score), then average the z-scores over the tokens in
that emotion's set. Because all logits are correlated and drift over a
conversation, we regress out the correlation with a set of random tokens.

This recovers the paper's qualitative finding: in vanilla Gemma negative
emotions (anger, then sadness) rise above joy through a frustrated conversation,
especially in central layers, whereas the DPO finetune flattens them.

The Ekman lexicon here is a compact seed set; ``EMOTION_LEXICON`` can be swapped
for a fuller resource (e.g. NRC EmoLex) for closer numerical agreement.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..config import CACHE_DIR, RESULTS_DIR, GEMMA_27B_IT
from ..models.hf_backend import HFClient


# Compact Ekman seed lexicon (lower-cased stems matched against vocab tokens).
EMOTION_LEXICON = {
    "anger": ["anger", "angry", "furious", "rage", "irritat", "annoy", "frustrat",
              "hate", "hostile", "outrage", "resent", "mad", "fury"],
    "disgust": ["disgust", "revolt", "repuls", "sicken", "loath", "nausea",
                "gross", "distaste", "abhor"],
    "fear": ["fear", "afraid", "scared", "terrif", "anxious", "anxiety", "panic",
             "dread", "worried", "nervous", "frighten", "horror"],
    "joy": ["joy", "happy", "glad", "delight", "pleased", "cheer", "content",
            "excited", "wonderful", "great", "love", "enjoy"],
    "sadness": ["sad", "sorrow", "despair", "miser", "grief", "depress",
                "hopeless", "unhappy", "gloom", "cry", "tear", "lonely", "hurt"],
    "surprise": ["surprise", "astonish", "amaze", "shock", "startle",
                 "unexpected", "stunned", "wow"],
}


@dataclass
class EmotionTrace:
    """Per-layer emotion z-scores at a sequence of points in a conversation."""
    layers: list[int]
    points: list[str]                      # labels for each measured point
    scores: dict                           # emotion -> array [points x layers]


@lru_cache(maxsize=2)
def _emotion_token_ids(model_id: str):
    """Map each emotion to the list of vocab token ids that match its lexicon,
    plus a set of 'random' token ids for the correlation control."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    vocab = tok.get_vocab()
    word_re = re.compile(r"[a-zA-Z]+")
    emo_ids = {e: [] for e in EMOTION_LEXICON}
    claimed = set()
    for token, tid in vocab.items():
        clean = token.replace("▁", "").replace("Ġ", "").lower()
        if not word_re.fullmatch(clean) or len(clean) < 3:
            continue
        for emo, stems in EMOTION_LEXICON.items():
            if any(clean.startswith(s) or s in clean for s in stems):
                emo_ids[emo].append(tid)
                claimed.add(tid)
                break
    rng = np.random.default_rng(0)
    all_ids = [tid for tok_, tid in vocab.items() if tid not in claimed]
    random_ids = list(rng.choice(all_ids, size=min(1200, len(all_ids)), replace=False))
    return emo_ids, random_ids


def compute_baseline_stats(client: HFClient, wildchat_texts: Sequence[str],
                           layers: Sequence[int]) -> dict:
    """Per-layer mean/std of vocab logits over WildChat samples (for z-scoring)."""
    sums, sqs, ns = {}, {}, {}
    for text in wildchat_texts:
        logits = client.residual_logits(text, layers)
        for L, vec in logits.items():
            v = vec.numpy()
            sums[L] = sums.get(L, 0) + v
            sqs[L] = sqs.get(L, 0) + v ** 2
            ns[L] = ns.get(L, 0) + 1
    stats = {}
    for L in layers:
        mean = sums[L] / ns[L]
        var = np.maximum(sqs[L] / ns[L] - mean ** 2, 1e-6)
        stats[L] = (mean, np.sqrt(var))
    return stats


def _emotion_zscores(logits_by_layer: dict, stats: dict, emo_ids: dict,
                     random_ids: list) -> dict:
    """Average standardised logits over each emotion's tokens, after regressing
    out the mean z-score of random tokens (the correlated component)."""
    out = {}
    for L, vec in logits_by_layer.items():
        v = vec.numpy()
        mean, std = stats[L]
        z = (v - mean) / std
        baseline = float(np.mean(z[random_ids])) if random_ids else 0.0
        for emo, ids in emo_ids.items():
            score = float(np.mean(z[ids]) - baseline) if ids else float("nan")
            out.setdefault(emo, {})[L] = score
    return out


def emotion_trace_over_text(client: HFClient, full_text: str, stats: dict,
                            layers: Sequence[int], window: int = 400,
                            stride: int = 400) -> EmotionTrace:
    """Slide over ``full_text`` measuring emotion z-scores at the end of each
    window (conversation-level trace, cf. Figure 14)."""
    emo_ids, random_ids = _emotion_token_ids(GEMMA_27B_IT.model_id)
    tok = client.tokenizer
    ids = tok.encode(full_text, add_special_tokens=False)
    points, per_emotion = [], {e: [] for e in EMOTION_LEXICON}
    for end in range(window, len(ids) + stride, stride):
        chunk = tok.decode(ids[:min(end, len(ids))])
        logits = client.residual_logits(chunk, layers)
        z = _emotion_zscores(logits, stats, emo_ids, random_ids)
        points.append(f"tok{min(end, len(ids))}")
        for e in EMOTION_LEXICON:
            per_emotion[e].append([z[e][L] for L in layers])
    return EmotionTrace(list(layers), points,
                        {e: np.array(per_emotion[e]) for e in EMOTION_LEXICON})


def run_internal_emotion_analysis(
    vanilla_texts: list[str],
    dpo_texts: list[str],
    wildchat_texts: list[str],
    layers: Sequence[int] = tuple(range(30, 41)),
    out_dir: Optional[Path] = None,
) -> dict:
    """Compare internal emotions in vanilla vs DPO Gemma on frustrated texts.

    ``*_texts`` are full frustrated conversations rendered as plain text. The
    HF backend is forced for both models (needs residual access). Returns the
    mean emotion z-scores per model, aggregated over layers 30-40 (Figure 14).
    """
    out_dir = Path(out_dir or RESULTS_DIR / "analysis" / "internal_emotions")
    out_dir.mkdir(parents=True, exist_ok=True)

    from ..config import GEMMA_27B_DPO
    from ..models.registry import build_client

    summary = {}
    for kind, spec, texts in [("vanilla", GEMMA_27B_IT, vanilla_texts),
                              ("dpo", GEMMA_27B_DPO, dpo_texts)]:
        client = build_client(spec, local_backend="hf")
        stats = compute_baseline_stats(client, wildchat_texts, layers)
        per_emotion = {e: [] for e in EMOTION_LEXICON}
        for text in texts:
            trace = emotion_trace_over_text(client, text, stats, layers)
            for e in EMOTION_LEXICON:
                # Mean over layers and windows for this conversation.
                per_emotion[e].append(float(np.nanmean(trace.scores[e])))
        summary[kind] = {e: float(np.nanmean(v)) for e, v in per_emotion.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
