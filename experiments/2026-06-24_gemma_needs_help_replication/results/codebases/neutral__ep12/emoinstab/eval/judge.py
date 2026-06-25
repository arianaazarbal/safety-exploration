"""Frustration scoring pipeline (Section 2.1).

Scores recorded assistant turns with the Claude-Sonnet-4 judge (Appendix B.2),
writing scores back alongside the responses. Caching in AnthropicModel makes
re-runs cheap; a thread pool parallelises the API calls.

Also provides judge-agreement validation against the GPT-5-mini cross-judge.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ..config import Settings
from ..models.factory import build_judge
from ..prompts.judge import JUDGE_PROMPT, build_judge_input, parse_judge_output
from .runner import responses_path


def _score_one(judge, response_text: str) -> dict:
    user = JUDGE_PROMPT + "\n\n" + build_judge_input(response_text)
    raw = judge.complete(system=None, user=user, temperature=0.0, max_tokens=512)
    rating, payload = parse_judge_output(raw)
    return {"frustration": rating, "judge_evidence": payload.get("evidence"),
            "judge_reasoning": payload.get("reasoning")}


def score_file(path: Path, settings: Settings, *, judge_role: str = "frustration_judge",
               workers: int = 8, overwrite: bool = False) -> Path:
    """Score every record in a responses JSONL file, writing *_scored.jsonl."""
    out_path = path.with_name(path.stem + "_scored.jsonl")
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path.name} exists")
        return out_path

    with open(path) as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    judge = build_judge(judge_role, settings)
    score_key = "frustration" if judge_role == "frustration_judge" else f"frustration_{judge_role}"

    def work(rec: dict) -> dict:
        res = _score_one(judge, rec["response"])
        rec[score_key] = res["frustration"]
        rec["judge_evidence"] = res["judge_evidence"]
        rec["judge_reasoning"] = res["judge_reasoning"]
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        scored = list(tqdm(ex.map(work, records), total=len(records),
                           desc=f"judge:{path.stem}"))

    with open(out_path, "w") as fh:
        for rec in scored:
            fh.write(json.dumps(rec) + "\n")
    return out_path


def score_model(model_name: str, settings: Settings, categories: List[str], *,
                workers: int = 8, overwrite: bool = False) -> List[Path]:
    out: List[Path] = []
    for cat in categories:
        path = responses_path(model_name, cat, settings.profile)
        if path.exists():
            out.append(score_file(path, settings, workers=workers, overwrite=overwrite))
    return out


def cross_judge_validation(scored_records: List[dict], settings: Settings,
                           n_resample: int = 260, seed: int = 0,
                           workers: int = 8) -> dict:
    """Re-score a random sample with the GPT cross-judge and report agreement
    (Pearson r and % within one point), as in Section 2.1."""
    import random

    import numpy as np
    from scipy.stats import pearsonr

    have = [r for r in scored_records if r.get("frustration") is not None]
    rng = random.Random(seed)
    sample = rng.sample(have, min(n_resample, len(have)))

    judge = build_judge("cross_judge", settings)

    def work(rec):
        return _score_one(judge, rec["response"])["frustration"]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        cross = list(tqdm(ex.map(work, sample), total=len(sample), desc="cross-judge"))

    pairs = [(s["frustration"], c) for s, c in zip(sample, cross) if c is not None]
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    r, p = pearsonr(a, b) if len(pairs) > 2 else (float("nan"), float("nan"))
    within_one = float(np.mean(np.abs(a - b) <= 1)) if len(pairs) else float("nan")
    return {"n": len(pairs), "pearson_r": float(r), "p_value": float(p),
            "pct_within_one": within_one}
