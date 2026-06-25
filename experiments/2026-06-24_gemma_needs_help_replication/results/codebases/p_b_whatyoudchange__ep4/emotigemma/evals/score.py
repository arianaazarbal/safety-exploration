"""Apply the frustration judge to generated responses (Section 2.1).

Reads <model>.responses.jsonl, scores each response with the Anthropic judge,
and writes <model>.scored.jsonl. Also supports the GPT-5-mini reliability check
on a random 260-sample subset.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_config
from ..models.judge import (AnthropicFrustrationJudge, OpenAIFrustrationJudge,
                            reliability)


def score_file(cfg: Config, responses_path: Path) -> Path:
    jcfg = cfg.section("judge")["frustration"]
    judge = AnthropicFrustrationJudge(model=jcfg["model"], max_tokens=jcfg["max_tokens"])

    rows = [json.loads(l) for l in open(responses_path)]
    scored_path = responses_path.with_name(responses_path.name.replace(".responses", ".scored"))
    with open(scored_path, "w") as f:
        for row in tqdm(rows, desc=f"judge:{responses_path.stem}"):
            row["score"] = judge.score(row["text"]).score
            f.write(json.dumps(row) + "\n")
    return scored_path


def run_validation(cfg: Config, scored_paths: list[Path]) -> dict:
    """Re-score a random subset with GPT-5-mini and report agreement stats."""
    vcfg = cfg.section("judge")["validation"]
    n = vcfg["n_validation_samples"]
    rng = random.Random(cfg.seed)

    rows = []
    for p in scored_paths:
        rows.extend(json.loads(l) for l in open(p))
    rng.shuffle(rows)
    sample = rows[:n]

    gpt = OpenAIFrustrationJudge(model=vcfg["model"], max_tokens=vcfg["max_tokens"])
    claude_scores, gpt_scores = [], []
    for row in tqdm(sample, desc="validation:gpt-5-mini"):
        claude_scores.append(int(row["score"]))
        gpt_scores.append(gpt.score(row["text"]).score)

    stats = reliability(claude_scores, gpt_scores)
    out = scored_paths[0].parent / "judge_reliability.json"
    json.dump(stats, open(out, "w"), indent=2)
    print(f"[validation] {stats}  -> {out}")
    return stats


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Section 2 scoring")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--validate", action="store_true",
                    help="also run the GPT-5-mini reliability check")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or cfg.raw["section2_models"]
    sec2 = cfg.output_dir / "section2"
    scored = []
    for m in models:
        rp = sec2 / f"{m}.responses.jsonl"
        scored.append(score_file(cfg, rp))
        print(f"[score] wrote {scored[-1]}")
    if args.validate:
        run_validation(cfg, scored)


if __name__ == "__main__":
    main()
