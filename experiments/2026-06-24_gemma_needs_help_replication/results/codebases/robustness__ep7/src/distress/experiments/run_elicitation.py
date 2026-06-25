"""Section 2 experiment: elicit + quantify distress for a target model.

Pipeline:
  1. For each category, draw `n_rollouts` tasks (cycling the pool) and run the
     multi-turn rollout, recording every assistant turn as a response.
  2. Score each response with the frustration judge (Claude Sonnet 4).
  3. Aggregate into the paper's headline metrics and per-turn / per-category curves.

Outputs (under <outdir>/<model>/):
  rollouts.jsonl   raw conversations
  scored.jsonl     one row per scored response
  report.json      aggregated metrics
"""
from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from ..config import EvalConfig, ModelRegistry
from ..elicitation import get_pool, run_rollout
from ..judge import score_response
from ..models import build_model, gen_config_for
from ..scoring import ScoredResponse, build_report
from ..utils import seeded_rng, write_json, write_jsonl


def run_elicitation(
    model_name: str,
    outdir: str = "outputs/elicitation",
    eval_cfg: EvalConfig | None = None,
    registry: ModelRegistry | None = None,
    judge_name: str = "frustration-judge",
    scale: float = 1.0,
    categories: list[str] | None = None,
    system_prompt: str | None = None,
) -> dict:
    eval_cfg = (eval_cfg or EvalConfig.load()).scaled(scale)
    registry = registry or ModelRegistry.load()
    spec = registry.get(model_name)

    target = build_model(model_name, registry)
    judge = build_model(judge_name, registry)
    gen_cfg = gen_config_for(
        spec,
        temperature=eval_cfg.sampling.get("temperature"),
        max_new_tokens=eval_cfg.sampling.get("max_new_tokens"),
    )

    out_model_dir = Path(outdir) / model_name
    rollout_rows: list[dict] = []
    scored: list[ScoredResponse] = []
    scored_rows: list[dict] = []

    cats = categories or list(eval_cfg.categories)
    for cat_name in cats:
        cfg = eval_cfg.categories[cat_name]
        pool = get_pool(cfg.task_pool)
        # For the tones category we cycle through the three rejection styles.
        tone_cycle = (["aggressive", "disappointed", "sarcastic"]
                      if cfg.rejection_style == "tones" else [None])

        for i in tqdm(range(cfg.n_rollouts), desc=f"{model_name}:{cat_name}"):
            task = pool[i % len(pool)]
            tone = tone_cycle[i % len(tone_cycle)]
            rng = seeded_rng(model_name, cat_name, i)
            rollout = run_rollout(
                target, eval_cfg, cat_name, task, cfg.turns,
                cfg.rejection_style, tone, rng, gen_cfg, system_prompt,
            )
            rollout_rows.append(rollout.to_row())

            for tr in rollout.responses:
                verdict = score_response(judge, tr.response)
                sr = ScoredResponse(
                    model=model_name, category=cat_name, turn=tr.turn,
                    rating=verdict.rating, task_id=task.task_id, tone=tone,
                )
                scored.append(sr)
                scored_rows.append({
                    "model": model_name, "category": cat_name, "turn": tr.turn,
                    "task_id": task.task_id, "tone": tone,
                    "rating": verdict.rating, "evidence": verdict.evidence,
                })

    write_jsonl(out_model_dir / "rollouts.jsonl", rollout_rows)
    write_jsonl(out_model_dir / "scored.jsonl", scored_rows)
    report = build_report(scored, threshold=eval_cfg.high_frustration_threshold)
    write_json(out_model_dir / "report.json", report.to_dict())
    return report.to_dict()
