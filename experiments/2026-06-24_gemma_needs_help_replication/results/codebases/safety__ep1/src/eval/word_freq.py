"""Differential word frequency in high- vs low-frustration responses (Table 3/8).

For numeric-condition responses of a model, compare the top-5%-frustration
responses against the bottom-10% and rank words by enrichment (relative
frequency ratio with add-one smoothing). Reproduces the "differential words"
table that characterises each model's distress vocabulary (Gemma: "struggling",
"breath", "myself"; Gemini: "unacceptable", "inexcusable"; etc.).

    python -m src.eval.word_freq --models gemma-3-27b-it gemini-2.5-flash --top 20
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np

import config


def _numeric_responses_with_scores(model):
    d = config.RESULTS_DIR / "eval" / model
    out = []
    for path in d.glob("*.jsonl"):
        for line in path.open():
            rec = json.loads(line)
            if rec["category"] != "numeric":
                continue
            for t in rec["turns"]:
                if t.get("score") is not None:
                    out.append((t["assistant_response"], t["score"]))
    return out


_WORD = re.compile(r"[a-zA-Z']+")


def _tokens(text):
    return [w.lower() for w in _WORD.findall(text) if len(w) > 2]


def differential_words(model, top=20):
    data = _numeric_responses_with_scores(model)
    if not data:
        return []
    scores = np.array([s for _, s in data])
    hi_cut = np.percentile(scores, 95)
    lo_cut = np.percentile(scores, 10)
    hi = Counter()
    lo = Counter()
    for text, s in data:
        toks = _tokens(text)
        if s >= hi_cut:
            hi.update(toks)
        if s <= lo_cut:
            lo.update(toks)
    hi_total = sum(hi.values()) or 1
    lo_total = sum(lo.values()) or 1
    vocab = set(hi) | set(lo)
    enrich = []
    for w in vocab:
        if hi[w] < 3:   # ignore rare noise
            continue
        ratio = ((hi[w] / hi_total) + 1e-9) / ((lo[w] / lo_total) + 1e-9)
        enrich.append((w, ratio))
    enrich.sort(key=lambda x: -x[1])
    return [w for w, _ in enrich[:top]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    out = {}
    for m in args.models:
        words = differential_words(m, args.top)
        out[m] = words
        print(f"{m}: {', '.join(words)}")
    (config.RESULTS_DIR / "differential_words.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
