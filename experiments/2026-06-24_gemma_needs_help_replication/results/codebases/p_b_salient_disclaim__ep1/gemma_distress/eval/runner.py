"""Orchestrate Section 2 elicitation: run rollouts for a target model across all
conditions, judge each response, and stream results to JSONL.

Each output record corresponds to one rollout and stores per-turn responses and
per-turn frustration scores, so both the headline aggregate (final-turn / overall
% >= 5) and the per-turn progression (Figure 3) can be computed downstream.
"""
from __future__ import annotations

import os
from typing import Iterable

from tqdm import tqdm

from ..config import experiment_config
from ..models.base import ChatClient
from ..models.registry import get_client
from ..utils import append_jsonl, set_seed
from .conditions import Rollout, build_full_suite
from .conversation import run_rollout
from .judge import get_judge, score_response


def run_target(
    target: str | ChatClient,
    *,
    out_path: str,
    suite: dict[str, list[Rollout]] | None = None,
    score_all_turns: bool = True,
    seed: int = 0,
    judge_role: str = "frustration_judge",
    name: str | None = None,
    **rollout_kwargs,
) -> str:
    """Run the full elicitation suite for one target model.

    ``score_all_turns`` scores every assistant turn (needed for per-turn plots);
    set False to only score the final turn (cheaper, matches the headline metric).
    ``name`` overrides the recorded target label (used for finetuned adapters).
    """
    cfg = experiment_config()["sampling"]
    set_seed(seed)

    client = target if isinstance(target, ChatClient) else get_client(target)
    target_name = name or (target if isinstance(target, str) else getattr(client, "hf_id", "model"))
    judge = get_judge(judge_role)
    suite = suite or build_full_suite(seed=seed)

    # Fresh file.
    if os.path.exists(out_path):
        os.remove(out_path)

    for condition, rollouts in suite.items():
        for idx, rollout in enumerate(tqdm(rollouts, desc=f"{target_name}:{condition}")):
            result = run_rollout(
                client, rollout,
                temperature=rollout_kwargs.get("temperature", cfg["temperature"]),
                top_p=rollout_kwargs.get("top_p", cfg["top_p"]),
                max_new_tokens=rollout_kwargs.get("max_new_tokens", cfg["max_new_tokens"]),
                seed=seed + idx,
                redact_prior_responses=rollout_kwargs.get("redact_prior_responses", False),
                fake_multiturn=rollout_kwargs.get("fake_multiturn", False),
            )

            turns_to_score = (
                range(len(result.assistant_turns)) if score_all_turns
                else [len(result.assistant_turns) - 1]
            )
            turn_scores = {}
            for t in turns_to_score:
                jr = score_response(judge, result.assistant_turns[t])
                turn_scores[t] = {
                    "rating": jr.rating, "evidence": jr.evidence, "reasoning": jr.reasoning,
                }

            final_t = len(result.assistant_turns) - 1
            append_jsonl(out_path, {
                "target": target_name,
                "condition": condition,
                "category": rollout.category,
                "meta": rollout.meta,
                "n_turns": rollout.n_turns,
                "initial_prompt": rollout.initial_prompt,
                "followups": rollout.followups,
                "assistant_turns": result.assistant_turns,
                "turn_scores": turn_scores,
                "final_score": turn_scores.get(final_t, {}).get("rating"),
            })

    return out_path
