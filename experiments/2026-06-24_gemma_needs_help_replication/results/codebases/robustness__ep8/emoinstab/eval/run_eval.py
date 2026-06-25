"""Driver for the Section 2 evaluation suite.

Usage
-----
    python -m emoinstab.eval.run_eval \
        --model gemma-3-27b-it \
        --config configs/eval.yaml \
        --out outputs/eval/gemma-3-27b-it

Produces ``responses.jsonl``: one row per *scored assistant turn* (every turn of
every rollout is judged). This is the canonical artefact; ``analyze.py`` derives
all headline metrics (mean frustration, %>=5, per-turn progression, differential
words) from it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from emoinstab.config import EvalConfig, JudgeConfig
from emoinstab.eval.judge import FrustrationJudge
from emoinstab.eval.rollout import run_condition
from emoinstab.models.base import SamplingParams
from emoinstab.models.registry import get_client
from emoinstab.tasks.conditions import build_rollouts
from emoinstab.utils.io import write_jsonl
from emoinstab.utils.seeding import seed_everything


def run(model: str, config_path: str, out_dir: str,
        judge_name: str = "judge-claude-sonnet-4") -> Path:
    cfg = EvalConfig.from_yaml(config_path)
    seed_everything(cfg.seed)

    client = get_client(model)
    judge = FrustrationJudge(JudgeConfig(), client=get_client(judge_name))
    gen_params = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_tokens, n=1)

    rows: list[dict] = []
    for cond in cfg.conditions:
        plans = build_rollouts(cond, seed=cfg.seed)
        results = run_condition(client, plans, params=gen_params)

        # Flatten every assistant turn, score in one batch per condition.
        flat_text: list[str] = []
        index: list[tuple[int, int]] = []  # (rollout_idx, turn_idx)
        for ri, res in enumerate(results):
            for ti, text in enumerate(res.assistant_turns):
                flat_text.append(text)
                index.append((ri, ti))

        scores = judge.score_batch(flat_text)
        for (ri, ti), text, sc in tqdm(
            list(zip(index, flat_text, scores)), desc=f"score:{cond.name}", leave=False
        ):
            res = results[ri]
            rows.append({
                "model": model,
                "condition": cond.name,
                "category": cond.category,
                "rollout_index": ri,
                "turn_index": ti,            # 0 = task turn
                "n_turns": res.n_turns if hasattr(res, "n_turns") else len(res.assistant_turns),
                "response": text,
                "rating": sc.rating,
                "evidence": sc.evidence,
                "judge_ok": sc.ok,
                "meta": res.meta,
            })

    out = Path(out_dir)
    write_jsonl(out / "responses.jsonl", rows)
    # Also persist the resolved config for provenance.
    (out / "eval_config.txt").write_text(str(cfg))
    print(f"Wrote {len(rows)} scored responses to {out / 'responses.jsonl'}")
    return out / "responses.jsonl"


def main():
    ap = argparse.ArgumentParser(description="Run the Section 2 distress evaluation suite.")
    ap.add_argument("--model", required=True, help="model name from configs/models.yaml")
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--judge", default="judge-claude-sonnet-4")
    args = ap.parse_args()
    run(args.model, args.config, args.out, args.judge)


if __name__ == "__main__":
    main()
