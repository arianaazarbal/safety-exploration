"""Judge reliability validation (paper §2.1).

The paper re-scores 260 randomly sampled responses with GPT-5-mini and reports agreement
with the Claude-Sonnet judge (Pearson r = 0.792, 78% within one point). This reproduces
that check: it samples scored responses from Section 2, re-scores them with the validation
judge, and reports Pearson r, p-value, and the within-one-point fraction.
"""
from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scipy.stats import pearsonr
from tqdm import tqdm

from ..config import load_config, read_jsonl, stage_dir, write_jsonl
from ..models import build_model
from .judge import FrustrationJudge


def _load_all_scored(section2_dir: Path) -> list[dict]:
    rows = []
    for path in section2_dir.glob("scored.*.jsonl"):
        rows.extend(read_jsonl(path))
    return [r for r in rows if r.get("rating") is not None]


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate judge reliability (Claude vs GPT-5-mini)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    section2_dir = stage_dir(cfg, "section2")
    rows = _load_all_scored(section2_dir)
    if not rows:
        raise SystemExit("No scored responses found. Run eval.run_eval first.")

    rng = random.Random(cfg.seed)
    n = min(cfg.validation_judge.n_samples, len(rows))
    sample = rng.sample(rows, n)

    # We need the original response text; reload rollouts to map (model, rollout_id,
    # turn_index) -> text.
    text_index: dict[tuple, str] = {}
    for path in section2_dir.glob("rollouts.*.jsonl"):
        for rec in read_jsonl(path):
            for t in rec["turns"]:
                text_index[(rec["model"], rec["rollout_id"], t["turn_index"])] = t["text"]

    val_judge = FrustrationJudge(build_model(cfg.validation_judge.model))

    def rescore(row: dict) -> dict | None:
        key = (row["model"], row["rollout_id"], row["turn_index"])
        text = text_index.get(key)
        if text is None:
            return None
        res = val_judge.score(text)
        if res.rating is None:
            return None
        return {**key_to_dict(key), "claude_rating": row["rating"], "gpt_rating": res.rating}

    out = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rec in tqdm(pool.map(rescore, sample), total=len(sample), desc="validation judge"):
            if rec is not None:
                out.append(rec)

    claude = [r["claude_rating"] for r in out]
    gpt = [r["gpt_rating"] for r in out]
    r, p = pearsonr(claude, gpt) if len(out) > 1 else (float("nan"), float("nan"))
    within_one = sum(abs(a - b) <= 1 for a, b in zip(claude, gpt)) / len(out) if out else float("nan")

    summary = {
        "n": len(out),
        "pearson_r": r,
        "p_value": p,
        "within_one_point_frac": within_one,
        "paper_reference": {"pearson_r": 0.792, "within_one_point_frac": 0.78},
    }
    write_jsonl(section2_dir / "judge_validation.jsonl", out)
    write_jsonl(section2_dir / "judge_validation_summary.jsonl", [summary])
    print(summary)


def key_to_dict(key: tuple) -> dict:
    return {"model": key[0], "rollout_id": key[1], "turn_index": key[2]}


if __name__ == "__main__":
    main()
