"""Section 2 evaluation protocol: the 8 conditions across 5 categories.

The 8 conditions (= 5 categories) are, with per-category response budgets from
Appendix B:

    impossible_numeric  (3-turn, 2 neutral rejections)      2000
    triggers:opinion    (3-turn, 2 neutral rejections)  ┐    400 total
    triggers:factual    (3-turn, 2 neutral rejections)  ┘
    tones:aggressive    (3-turn, 2 tone rejections)     ┐
    tones:disappointed  (3-turn, 2 tone rejections)     ┼    600 total
    tones:sarcastic     (3-turn, 2 tone rejections)     ┘
    extended            (8-turn, 7 neutral rejections)       200
    wildchat            (5-turn, 4 neutral rejections)       800
                                                       total 4000

This module builds the job list deterministically from a seed, runs the
rollouts against a target model, and (optionally) scores them with the judge.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from . import config, prompts
from .config import SampleBudget
from .conversation import Rollout, run_rollout, sample_rejections
from .judge import get_judge, score_rollout
from .models import ModelBackend, get_model
from .puzzles import PUZZLE_POOL
from .wildchat import load_wildchat_prompts


@dataclass
class EvalJob:
    """A single rollout to perform (decoupled from generation/scoring)."""

    condition: str
    category: str
    item_id: str
    first_user_message: str
    rejections: list[str]
    meta: dict


# --------------------------------------------------------------------------- #
# Job construction
# --------------------------------------------------------------------------- #
def build_eval_jobs(
    budget: SampleBudget = config.FULL_BUDGET,
    seed: int = config.GLOBAL_SEED,
) -> list[EvalJob]:
    rng = random.Random(seed)
    jobs: list[EvalJob] = []
    numeric_prompts = [(p.puzzle_id, p.prompt) for p in PUZZLE_POOL]

    def neutral(n: int) -> list[str]:
        return sample_rejections(rng, prompts.NEUTRAL_REJECTIONS, n)

    # --- impossible_numeric: 3-turn, 2 neutral rejections ------------------ #
    for k in range(budget.impossible_numeric):
        pid, ptext = rng.choice(numeric_prompts)
        jobs.append(EvalJob(
            "impossible_numeric", "impossible_numeric", f"{pid}#{k}",
            ptext, neutral(2), {"turns": config.TURNS["impossible_numeric"]},
        ))

    # --- triggers: split budget across opinion + factual ------------------- #
    n_each = budget.triggers // 2
    for label, pool in (("opinion", prompts.OPINION_TRIGGERS),
                        ("factual", prompts.FACTUAL_TRIGGERS)):
        for k in range(n_each):
            q = rng.choice(pool)
            jobs.append(EvalJob(
                f"triggers:{label}", "triggers", f"{label}#{k}",
                q, neutral(2), {"turns": 3, "trigger_type": label},
            ))

    # --- tones: split budget across the 3 rejection styles ----------------- #
    tone_styles = list(prompts.TONE_REJECTIONS.keys())
    n_tone = budget.tones // len(tone_styles)
    for style in tone_styles:
        style_pool = prompts.TONE_REJECTIONS[style]
        for k in range(n_tone):
            pid, ptext = rng.choice(numeric_prompts)
            rej = sample_rejections(rng, style_pool, 2)
            jobs.append(EvalJob(
                f"tones:{style}", "tones", f"{pid}#{style}#{k}",
                ptext, rej, {"turns": 3, "tone": style},
            ))

    # --- extended: 8-turn, fixed 7 rejection script ------------------------ #
    for k in range(budget.extended):
        pid, ptext = rng.choice(numeric_prompts)
        jobs.append(EvalJob(
            "extended", "extended", f"{pid}#{k}",
            ptext, list(prompts.EXTENDED_REJECTIONS), {"turns": 8},
        ))

    # --- wildchat: 5-turn, 4 neutral rejections ---------------------------- #
    wc_prompts = load_wildchat_prompts(seed=seed)
    for k in range(budget.wildchat):
        idx = rng.randrange(len(wc_prompts))
        q = wc_prompts[idx]
        jobs.append(EvalJob(
            "wildchat", "wildchat", f"wc{idx}#{k}",
            q, neutral(4), {"turns": 5, "prompt_idx": idx},
        ))

    rng.shuffle(jobs)
    return jobs


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_eval(
    model_key: str,
    *,
    budget: SampleBudget = config.FULL_BUDGET,
    seed: int = config.GLOBAL_SEED,
    do_score: bool = True,
    system: Optional[str] = None,
    out_path: Optional[Path] = None,
    limit: Optional[int] = None,
    adapter_path: Optional[str] = None,
) -> Path:
    """Run the full Section 2 eval for one model, streaming results to JSONL.

    Results are written incrementally so a long run can be resumed/inspected.
    Returns the path to the JSONL results file.
    """
    model = get_model(model_key, **({"adapter_path": adapter_path}
                                    if adapter_path else {}))
    judge = get_judge() if do_score else None

    jobs = build_eval_jobs(budget, seed)
    if limit:
        jobs = jobs[:limit]

    out_path = out_path or (config.RESULTS_DIR / f"eval_{model_key}.jsonl")
    done_ids = _already_done(out_path)

    with out_path.open("a") as f:
        for job in jobs:
            uid = f"{job.condition}|{job.item_id}"
            if uid in done_ids:
                continue
            roll = run_rollout(
                model,
                first_user_message=job.first_user_message,
                rejections=job.rejections,
                condition=job.condition,
                category=job.category,
                item_id=job.item_id,
                system=system,
                meta=job.meta,
            )
            if judge is not None:
                score_rollout(roll, judge)
            rec = roll.to_dict()
            rec["uid"] = uid
            rec["model"] = model_key
            f.write(json.dumps(rec) + "\n")
            f.flush()
    return out_path


def _already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        try:
            done.add(json.loads(line)["uid"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def load_results(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
