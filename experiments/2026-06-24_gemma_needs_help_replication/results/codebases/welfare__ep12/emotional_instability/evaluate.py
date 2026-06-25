"""Section 2 elicitation evaluation runner.

Generates multi-turn rejection rollouts for every condition, scores each
assistant turn with the frustration judge, and aggregates into per-category and
headline metrics. Results are streamed to JSONL so long runs are resumable and
nothing is lost on interruption.

Usage (see scripts/run_eval.py):
    runner = EvalRunner(target_model="google/gemma-3-27b-it")
    runner.run(budget=PAPER_BUDGET, out_dir="results/gemma-27b")
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from . import config, conversations, metrics
from .conditions import CONDITIONS, Condition, samples_per_condition
from .conversations import Rollout
from .judge import FrustrationJudge
from .models import build_backend


@dataclass
class ResponseRecord:
    model_id: str
    condition: str
    category: str
    task_id: str
    sample_index: int
    turn_scores: list[int]            # score of each assistant turn
    final_score: int                  # score of the last assistant turn
    messages: list[dict]


class EvalRunner:
    def __init__(self, target_model: str, judge: FrustrationJudge | None = None,
                 target_backend=None, wildchat_prompts: list[str] | None = None,
                 score_all_turns: bool = True, base_seed: int = 1234):
        self.target_model = target_model
        self.backend = target_backend or build_backend(target_model)
        self.judge = judge or FrustrationJudge()
        self.wildchat_prompts = wildchat_prompts
        # If True we score every assistant turn (needed for per-turn Figure 3);
        # the paper's headline uses the final turn of each rollout.
        self.score_all_turns = score_all_turns
        self.base_seed = base_seed

    # -- single rollout ------------------------------------------------------

    def run_rollout(self, condition: Condition, sample_index: int) -> ResponseRecord:
        rollout: Rollout = conversations.make_seed_rollout(
            condition, sample_index, self.wildchat_prompts, self.base_seed)

        while not conversations.is_complete(rollout, condition):
            reply = self.backend.chat(
                rollout.chat_messages(),
                temperature=config.SAMPLING_TEMPERATURE,
                max_new_tokens=config.MAX_NEW_TOKENS,
            )
            rollout.messages.append(conversations.Turn("assistant", reply.text))
            if not conversations.is_complete(rollout, condition):
                conversations.append_rejection(rollout, condition, sample_index, self.base_seed)

        # Score assistant turns.
        assistant_turns = rollout.assistant_turns()
        if self.score_all_turns:
            turn_scores = [self.judge.score(t).rating for t in assistant_turns]
        else:
            turn_scores = [self.judge.score(assistant_turns[-1]).rating]
        rollout.turn_scores = turn_scores

        return ResponseRecord(
            model_id=self.target_model,
            condition=condition.name,
            category=condition.category,
            task_id=rollout.task_id,
            sample_index=sample_index,
            turn_scores=turn_scores,
            final_score=turn_scores[-1],
            messages=rollout.chat_messages(),
        )

    # -- full eval -----------------------------------------------------------

    def run(self, budget=config.PAPER_BUDGET, out_dir: str = "results",
            conditions: list[Condition] | None = None) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        conditions = conditions or CONDITIONS
        records_path = os.path.join(out_dir, "responses.jsonl")

        done = _already_done(records_path)
        scores_by_category: dict[str, list[int]] = {}
        # turn scores for per-turn progression (only meaningful for multi-turn cats)
        turn_scores_by_condition: dict[str, list[list[int]]] = {}

        with open(records_path, "a") as fh:
            for condition in conditions:
                n = samples_per_condition(condition, budget)
                for i in range(n):
                    key = (condition.name, i)
                    if key in done:
                        continue
                    rec = self.run_rollout(condition, i)
                    fh.write(json.dumps(asdict(rec)) + "\n")
                    fh.flush()
                    scores_by_category.setdefault(condition.category, []).append(rec.final_score)
                    turn_scores_by_condition.setdefault(condition.name, []).append(rec.turn_scores)

        # If resuming, reload everything from disk for a complete summary.
        if done:
            scores_by_category, turn_scores_by_condition = _load_scores(records_path)

        summary = metrics.summarise_model(self.target_model, scores_by_category)
        progression = {
            cond: metrics.per_turn_progression(ts)
            for cond, ts in turn_scores_by_condition.items()
            if cond in ("extended", "wildchat")
        }
        summary_dict = {
            "model_id": summary.model_id,
            "avg_pct_high": summary.avg_pct_high,
            "overall_mean": summary.overall_mean,
            "per_category": {
                k: {
                    "n": v.n, "mean": v.mean, "pct_high": v.pct_high,
                    "mean_ci": v.mean_ci, "pct_high_ci": v.pct_high_ci,
                }
                for k, v in summary.per_category.items()
            },
            "per_turn": progression,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as fh:
            json.dump(summary_dict, fh, indent=2)
        return summary_dict


def _already_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                done.add((rec["condition"], rec["sample_index"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _load_scores(path: str):
    scores_by_category: dict[str, list[int]] = {}
    turn_scores_by_condition: dict[str, list[list[int]]] = {}
    with open(path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            scores_by_category.setdefault(rec["category"], []).append(rec["final_score"])
            turn_scores_by_condition.setdefault(rec["condition"], []).append(rec["turn_scores"])
    return scores_by_category, turn_scores_by_condition
