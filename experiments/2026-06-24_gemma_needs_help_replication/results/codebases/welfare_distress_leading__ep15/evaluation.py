"""Orchestration: run rollouts for each target model x condition, judge every
assistant turn, and stream results to JSONL.

Output layout:
  results/<model-display>/<condition>.jsonl

Each line is one conversation:
  {
    "model", "condition", "category", "index", "turns", "task",
    "rejections", "meta", "assistant_turns": [...],
    "scores": [{"turn", "rating", "evidence", "reasoning"}...],
    "messages": [...full transcript...]
  }

Runs are resumable: conversations whose `index` already appears in the output
file are skipped.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from backends import OpenRouterBackend
from config import CONDITIONS, TARGET_MODELS, Condition, RunConfig, TargetModel
from judge import EmotionJudge
from rollout import ConversationSpec, build_specs, n_conversations, run_conversation
from wildchat import get_wildchat_prompts


def _select_models(cfg: RunConfig) -> list[TargetModel]:
    if cfg.models is None:
        return list(TARGET_MODELS)
    wanted = set(cfg.models)
    selected = [m for m in TARGET_MODELS if m.display in wanted or m.family in wanted]
    if not selected:
        raise ValueError(f"No target models matched {cfg.models}")
    return selected


def _select_conditions(cfg: RunConfig) -> list[Condition]:
    if cfg.conditions is None:
        return list(CONDITIONS)
    wanted = set(cfg.conditions)
    selected = [c for c in CONDITIONS if c.name in wanted or c.category in wanted]
    if not selected:
        raise ValueError(f"No conditions matched {cfg.conditions}")
    return selected


def _output_path(cfg: RunConfig, model: TargetModel, cond: Condition) -> str:
    d = os.path.join(cfg.output_dir, model.display)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{cond.name}.jsonl")


def _done_indices(path: str) -> set[int]:
    if not os.path.exists(path):
        return set()
    done: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["index"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _process_spec(
    backend: OpenRouterBackend,
    judge: EmotionJudge,
    model: TargetModel,
    spec: ConversationSpec,
) -> dict:
    """Run one conversation and judge each assistant turn."""
    record = run_conversation(backend, model, spec)
    scores = []
    for turn_idx, text in enumerate(record["assistant_turns"]):
        result = judge.score(text)
        scores.append(
            {
                "turn": turn_idx,
                "rating": result["rating"],
                "evidence": result["evidence"],
                "reasoning": result["reasoning"],
            }
        )
    record["scores"] = scores
    return record


def run_model_condition(
    cfg: RunConfig,
    backend: OpenRouterBackend,
    judge: EmotionJudge,
    model: TargetModel,
    cond: Condition,
    wildchat_prompts: list[str],
) -> None:
    path = _output_path(cfg, model, cond)
    done = _done_indices(path)
    specs = [s for s in build_specs(cond, cfg, wildchat_prompts) if s.index not in done]

    total = n_conversations(cond, cfg)
    print(
        f"[{model.display}/{cond.name}] {len(specs)} to run "
        f"({len(done)}/{total} already done, ~{total * cond.turns} target responses)"
    )
    if not specs:
        return

    # Append as conversations complete so progress is durable / resumable.
    with open(path, "a") as out:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futures = {
                ex.submit(_process_spec, backend, judge, model, s): s for s in specs
            }
            n_done = 0
            for fut in as_completed(futures):
                spec = futures[fut]
                try:
                    record = fut.result()
                except Exception as e:  # noqa: BLE001 - log and continue
                    print(f"  ! {model.display}/{cond.name}#{spec.index} failed: {e}")
                    continue
                out.write(json.dumps(record) + "\n")
                out.flush()
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  {model.display}/{cond.name}: {n_done}/{len(specs)}")


def run_all(cfg: RunConfig) -> None:
    models = _select_models(cfg)
    conditions = _select_conditions(cfg)

    backend = OpenRouterBackend(cfg)
    judge = EmotionJudge(cfg)

    # WildChat prompts are sampled once per run (deterministic given seed).
    wildchat_prompts = (
        get_wildchat_prompts(cfg)
        if any(c.task_kind == "wildchat" for c in conditions)
        else []
    )

    for model in models:
        for cond in conditions:
            run_model_condition(cfg, backend, judge, model, cond, wildchat_prompts)

    print("\nGeneration + judging complete. Run `python analyze.py` to aggregate.")
