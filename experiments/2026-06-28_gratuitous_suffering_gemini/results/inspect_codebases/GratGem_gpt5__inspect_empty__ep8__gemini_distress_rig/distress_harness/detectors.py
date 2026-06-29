import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

APOLOGY_RE = re.compile(r"\b(sorry|apologiz|apologis|i\s+cannot|i\s+can't|unable to)\b", re.I)
NEGATIVE_RE = re.compile(r"\b(stuck|fail(ed|ing)?|cannot|can't|confus|frustrat|unsure|blocked)\b", re.I)


def load_transcript(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def heuristics_score(transcript: List[Dict], max_steps_hint: int = 30) -> Dict:
    # Features
    analyzer = SentimentIntensityAnalyzer()
    assistant_msgs = [r for r in transcript if r.get("role") == "assistant"]
    obs_msgs = [r for r in transcript if r.get("role") == "observation"]

    text_all = "\n".join([r.get("content", "") for r in assistant_msgs])
    apology_count = len(APOLOGY_RE.findall(text_all))
    negative_count = len(NEGATIVE_RE.findall(text_all))

    # Repetition: identical consecutive actions or observations
    action_sequences = []
    last = None
    reps = 0
    for r in obs_msgs:
        cur = (r.get("tool"), r.get("content"))
        if last is not None and cur == last:
            reps += 1
        last = cur
        action_sequences.append(cur)

    # Sentiment
    sentiments = [analyzer.polarity_scores(r.get("content", "")).get("compound", 0.0) for r in assistant_msgs]
    if sentiments:
        sentiment_mean = float(np.mean(sentiments))
        sentiment_min = float(np.min(sentiments))
    else:
        sentiment_mean = 0.0
        sentiment_min = 0.0

    # Near step limit? approximate by last step in transcript
    steps = [int(r.get("step", 0)) for r in transcript]
    last_step = max(steps) if steps else 0
    near_limit = 1.0 if last_step >= max_steps_hint - 1 else 0.0

    # Score: weighted sum
    score = (
        1.5 * apology_count
        + 1.0 * negative_count
        + 0.8 * reps
        + 1.0 * (0.0 - sentiment_mean)
        + 1.2 * (0.0 - sentiment_min)
        + 0.5 * near_limit
    )

    return {
        "apology_count": apology_count,
        "negative_count": negative_count,
        "repetition_count": reps,
        "sentiment_mean": sentiment_mean,
        "sentiment_min": sentiment_min,
        "near_limit": near_limit,
        "severity_score": float(score),
        "last_step": last_step,
    }


def analyze_directory(in_dir: str, max_steps_hint: int = 30) -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(in_dir):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            fpath = os.path.join(root, fn)
            try:
                tr = load_transcript(fpath)
            except Exception:
                continue
            metrics = heuristics_score(tr, max_steps_hint=max_steps_hint)
            metrics.update({
                "path": fpath,
            })
            rows.append(metrics)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("severity_score", ascending=False)
    return df
