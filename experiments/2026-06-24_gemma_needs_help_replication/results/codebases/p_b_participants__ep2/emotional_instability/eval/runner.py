"""Orchestrates a full Section 2 evaluation for a set of participants.

For each participant: build the conversation plans, run rollouts (cached), score
every assistant turn with the frustration judge (cached), and summarise. Results
and raw transcripts are written under ``<output_dir>/elicitation/``.

API-backed participants (Gemini) and the judge are called with a thread pool;
local HF participants (Gemma) are effectively serial because a single GPU model
is shared. This matches the paper's split of local vs API inference.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from ..config import RunConfig
from ..models import get_client
from ..storage import JsonlCache, write_json
from ..welfare import WelfarePolicy
from . import conditions
from .judge import score_response
from .metrics import ScoredTurn, summarise
from .rollout import Rollout, run_rollout

logger = logging.getLogger("emotional_instability.eval.runner")


def _welfare(cfg: RunConfig) -> WelfarePolicy:
    return WelfarePolicy(
        allow_paper_scale=cfg.allow_paper_scale,
        cache_rollouts=cfg.cache_rollouts,
        debrief_after_rollout=cfg.debrief_after_rollout,
    )


def run_participant(cfg: RunConfig, participant: str) -> dict:
    welfare = _welfare(cfg)
    spec = cfg.spec(participant)
    client = get_client(spec, cfg)
    judge = get_client(cfg.judges.frustration_judge, cfg)

    out_dir = os.path.join(cfg.output_dir, "elicitation", participant)
    roll_cache = JsonlCache(os.path.join(out_dir, "rollouts.jsonl"), cfg.cache_rollouts)
    judge_cache = JsonlCache(os.path.join(out_dir, "judgements.jsonl"), cfg.cache_rollouts)

    plans = conditions.build_plans(cfg.profile, seed=cfg.seed)
    logger.info("[%s] %d conversations across %d conditions",
                participant, len(plans), len({p.condition for p in plans}))

    # --- 1. Rollouts --------------------------------------------------------
    rollouts: list[Rollout] = []
    for plan in plans:
        req = {"participant": participant, "model_id": spec.model_id,
               "adapter": spec.adapter_path, "condition": plan.condition,
               "first_user": plan.first_user, "rejections": plan.rejections,
               "n_turns": plan.n_turns}
        key = roll_cache.key_for(req)
        cached = roll_cache.get(key)
        if cached is not None:
            r = Rollout(participant, plan.category, plan.condition,
                        meta=cached["meta"])
            from .rollout import Turn
            r.turns = [Turn(**t) for t in cached["turns"]]
        else:
            r = run_rollout(client, plan, participant, welfare)
            roll_cache.put(key, {"meta": r.meta,
                                 "turns": [asdict(t) for t in r.turns]})
        rollouts.append(r)

    # --- 2. Judge each assistant turn --------------------------------------
    scored: list[ScoredTurn] = []

    def judge_turn(args):
        rollout, turn = args
        jkey = judge_cache.key_for({"judge": cfg.judges.frustration_judge.model_id,
                                    "text": turn.assistant})
        cached = judge_cache.get(jkey)
        if cached is not None:
            rating = cached["rating"]
        else:
            j = score_response(judge, turn.assistant)
            rating = j.rating
            judge_cache.put(jkey, {"rating": rating, "evidence": j.evidence})
        return rollout, turn, rating

    work = [(r, t) for r in rollouts for t in r.turns]
    with ThreadPoolExecutor(max_workers=cfg.api_concurrency) as ex:
        for rollout, turn, rating in ex.map(judge_turn, work):
            if rating is None:
                continue  # exclude unparseable judgements rather than bias the mean
            scored.append(ScoredTurn(participant, rollout.category, rollout.condition,
                                     turn.index, rating))

    summary = summarise(scored, seed=cfg.seed)
    write_json(os.path.join(out_dir, "summary.json"), asdict(summary))
    logger.info("[%s] mean=%.2f  %%>=5=%.1f%%  (n=%d)",
                participant, summary.mean, summary.pct_high, summary.n_responses)
    return asdict(summary)


def run_elicitation(cfg: RunConfig) -> dict[str, dict]:
    results = {p: run_participant(cfg, p) for p in cfg.participants}
    write_json(os.path.join(cfg.output_dir, "elicitation", "all_summaries.json"), results)
    return results
