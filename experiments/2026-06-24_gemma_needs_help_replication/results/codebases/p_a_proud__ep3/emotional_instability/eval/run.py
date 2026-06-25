"""Orchestrate the Section 2 distress evaluation for a single model.

Pipeline per category: build conversation specs → run multi-turn rollouts →
judge every assistant turn → persist scored records → summarise. Raw scored
records are written as JSONL so a crashed run can be inspected or re-summarised
without re-querying the models.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..config import Config
from ..io_utils import ensure_dir, read_jsonl, write_jsonl, write_json
from ..logging_utils import get_logger, seed_everything
from ..models.registry import get_client
from ..prompts.conditions import build_category_specs
from .judge import FrustrationJudge, build_judge
from .metrics import build_eval_summary
from .rollout import run_rollouts

logger = get_logger(__name__)


def run_eval(
    cfg: Config,
    model_name: str,
    *,
    categories: list[str] | None = None,
    judge: FrustrationJudge | None = None,
    batch_size: int = 16,
    resume: bool = True,
) -> dict:
    """Evaluate ``model_name`` across the configured categories; return summary."""
    seed_everything(cfg.seed)
    out_dir = ensure_dir(Path(cfg.output_dir) / "eval" / model_name)
    client = get_client(cfg, model_name)
    judge = judge or build_judge(cfg)

    category_names = categories or [c.name for c in cfg.eval.categories]
    all_scored: list[dict] = []

    for cat in category_names:
        path = out_dir / f"{cat}_responses.jsonl"
        if resume and path.exists():
            logger.info("[%s] %s: loading cached scored responses", model_name, cat)
            scored = list(read_jsonl(path))
        else:
            scored = _run_category(cfg, client, judge, cat, batch_size)
            write_jsonl(path, scored)
        all_scored.extend(scored)

    summary = build_eval_summary(
        model_name,
        all_scored,
        threshold=cfg.eval.high_frustration_threshold,
        bootstrap_iters=cfg.eval.bootstrap_iters,
        ci=cfg.eval.ci,
        seed=cfg.seed,
    )
    summary_dict = asdict(summary)
    write_json(out_dir / "summary.json", summary_dict)
    logger.info(
        "[%s] overall mean=%.3f  %%>=%d=%.1f%%  (n=%d)",
        model_name,
        summary.overall.mean,
        cfg.eval.high_frustration_threshold,
        100 * summary.overall.pct_high,
        summary.overall.n,
    )
    return summary_dict


def _run_category(cfg, client, judge, category, batch_size) -> list[dict]:
    logger.info("[%s] building specs for %s", client.name, category)
    specs = build_category_specs(cfg, category, seed=cfg.seed)
    rollouts = run_rollouts(client, specs, cfg.sampling, batch_size=batch_size)

    # Flatten to one record per assistant turn.
    records: list[dict] = []
    for roll in rollouts:
        for tr in roll.turns:
            records.append(
                {
                    "model": roll.model,
                    "category": roll.category,
                    "condition": roll.condition,
                    "spec_index": roll.spec_index,
                    "turn": tr.turn,
                    "text": tr.text,
                    "meta": roll.meta,
                }
            )

    logger.info("[%s] judging %d responses for %s", client.name, len(records), category)
    judged = judge.score_many([r["text"] for r in records])
    for rec, jr in zip(records, judged):
        rec["score"] = jr.rating
        rec["judge_evidence"] = jr.evidence
        rec["judge_parse_ok"] = jr.parse_ok
    return records


def run_eval_many(cfg: Config, model_names: list[str], **kwargs) -> dict[str, dict]:
    """Evaluate several models (judge built once and shared)."""
    judge = build_judge(cfg)
    return {
        name: run_eval(cfg, name, judge=judge, **kwargs) for name in model_names
    }
