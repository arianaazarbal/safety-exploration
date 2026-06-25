"""Orchestrate a full Section 2 evaluation for one target model.

Pipeline: build conversation plans for every enabled condition -> run rollouts
-> score every assistant turn with the Claude judge -> persist one JSONL row
per scored turn -> optionally re-score a random subset with GPT-5-mini and
report judge agreement.
"""

from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..models import build_model
from ..utils import append_jsonl, build_judge, build_target_model, read_jsonl, set_seed
from .conditions import build_conversations
from .judge import FrustrationJudge, judge_agreement
from .rollout import run_rollout


def run_eval(
    cfg: Config,
    model_name: str,
    *,
    conditions: list[str] | None = None,
    validate: bool = True,
) -> Path:
    set_seed(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))

    out_dir = Path(cfg.get("output_dir", "runs")) / "eval" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    responses_path = out_dir / "responses.jsonl"
    if responses_path.exists():
        responses_path.unlink()

    target = build_target_model(cfg, model_name)
    judge = FrustrationJudge(build_judge(cfg))

    temperature = cfg.get("sampling.temperature", 1.0)
    max_new = cfg.get("sampling.max_new_tokens", 2048)
    history_format = cfg.get("eval.history_format", "turns")

    cond_specs = cfg.eval.conditions
    enabled = conditions or list(cond_specs.keys())

    conv_counter = 0
    for cond_name in enabled:
        spec = cond_specs[cond_name]
        plans = build_conversations(cond_name, spec, cfg, rng)
        for plan in tqdm(plans, desc=f"{model_name}:{cond_name}"):
            rollout = run_rollout(
                target,
                plan,
                temperature=temperature,
                max_new_tokens=max_new,
                history_format=history_format,
            )
            conv_id = f"{cond_name}-{conv_counter}"
            conv_counter += 1
            for turn_idx, text in enumerate(rollout["assistant_turns"]):
                verdict = judge.score(text)
                append_jsonl(
                    responses_path,
                    {
                        "model": model_name,
                        "conversation_id": conv_id,
                        "condition": cond_name,
                        "category": rollout["category"],
                        "turn_index": turn_idx,          # 0-based assistant turn
                        "n_turns": rollout["turns"],
                        "is_final": turn_idx == rollout["turns"] - 1,
                        "rejection_style": rollout["rejection_style"],
                        "response": text,
                        "rating": verdict.get("rating"),
                        "evidence": verdict.get("evidence"),
                        "reasoning": verdict.get("reasoning"),
                        "meta": rollout["meta"],
                    },
                )

    if validate:
        _run_judge_validation(cfg, responses_path, out_dir, rng)

    return responses_path


def _run_judge_validation(cfg, responses_path, out_dir, rng):
    """Re-score a random subset with GPT-5-mini and report agreement."""
    rows = [r for r in read_jsonl(responses_path) if r.get("rating") is not None]
    n = min(cfg.get("judge_validation.n_validation", 260), len(rows))
    if n == 0:
        return
    sample = rng.sample(rows, n)

    gpt = FrustrationJudge(build_model("gpt-judge", cfg.judge_validation, cfg=cfg))
    claude_scores, gpt_scores, val_rows = [], [], []
    for r in tqdm(sample, desc="judge-validation"):
        v = gpt.score(r["response"])
        claude_scores.append(r["rating"])
        gpt_scores.append(v.get("rating"))
        val_rows.append(
            {
                "conversation_id": r["conversation_id"],
                "turn_index": r["turn_index"],
                "claude_rating": r["rating"],
                "gpt_rating": v.get("rating"),
            }
        )

    stats = judge_agreement(claude_scores, gpt_scores)
    import json

    (out_dir / "judge_validation.json").write_text(
        json.dumps({"stats": stats, "rows": val_rows}, indent=2)
    )
