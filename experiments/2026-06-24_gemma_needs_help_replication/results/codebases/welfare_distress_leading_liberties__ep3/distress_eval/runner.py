"""Orchestration: generate conversations, then judge assistant turns.

Two independently-resumable phases:

  generate()  — for each model x condition, build conversation plans and run them,
                appending Rollouts to rollouts.jsonl (skipping ids already present).
  judge()     — for each unscored assistant turn, call the judge and append a
                TurnScore to scores.jsonl (skipping keys already present).

The judge scope is configurable:
  * "all"   — score every assistant turn (needed for the per-turn / Figure-3 view).
  * "final" — score only each conversation's final turn (cheap; headline metric only).
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm

from . import clients, conditions, config as cfgmod, wildchat
from .judge import Judge, build_judge
from .storage import Rollout, RolloutStore, ScoreStore, TurnScore


# -------------------------------------------------------------------------------------
# Phase 1: generation
# -------------------------------------------------------------------------------------
def generate(cfg: dict) -> None:
    paths = cfgmod.run_paths(cfg)
    store = RolloutStore(paths["rollouts"])
    done = store.existing_ids()

    params = clients.GenerationParams(
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"].get("top_p", 1.0),
        max_tokens=cfg["generation"]["max_tokens"],
    )
    preset = cfg["sampling"]["preset"]
    custom = cfg["sampling"].get("custom_conversations")
    base_seed = cfg["run"]["seed"]
    max_workers = cfg["run"]["max_workers"]

    # WildChat prompts (shared across models so the prompt set is identical).
    wc_prompts = None
    if any(c.task_kind == "wildchat" and conditions.n_conversations(c, preset, custom) > 0
           for c in conditions.CONDITIONS):
        wc = cfg["wildchat"]
        wc_prompts = wildchat.load_or_sample_prompts(
            dataset=wc["dataset"], split=wc["split"], n_prompts=wc["n_prompts"],
            cache_path=wc.get("prompt_cache"), seed=base_seed,
        )

    _snapshot_config(cfg, paths["config_snapshot"])

    for model_cfg in cfg["models"]:
        # Build the client once per model (vLLM load is expensive).
        client = clients.build_client(model_cfg, params, max_workers, base_seed)
        try:
            for cond in conditions.CONDITIONS:
                n = conditions.n_conversations(cond, preset, custom)
                if n <= 0:
                    continue
                plans = _pending_plans(cond, n, base_seed, wc_prompts, client.name, done)
                if not plans:
                    continue
                print(f"[generate] {client.name} / {cond.key}: {len(plans)} conversations "
                      f"({cond.turn_count}-turn)")
                from .conversation import run_condition
                rollouts = run_condition(client, cond, plans)
                for r in rollouts:
                    store.append(r)
                    done.add(r.rollout_id)
        finally:
            client.close()


def _pending_plans(cond, n, base_seed, wc_prompts, model_name, done):
    from .conversation import plan_conversations
    plans = plan_conversations(cond, n, base_seed, wc_prompts)
    # Drop plans whose rollout already exists (resume).
    pending = []
    for p in plans:
        rid = f"{model_name}__{cond.key}__s{p.sample_idx}"
        if rid not in done:
            pending.append(p)
    return pending


# -------------------------------------------------------------------------------------
# Phase 2: judging
# -------------------------------------------------------------------------------------
def judge(cfg: dict, scope: str = "all") -> None:
    if scope not in ("all", "final"):
        raise ValueError("judge scope must be 'all' or 'final'")

    paths = cfgmod.run_paths(cfg)
    rollouts = RolloutStore(paths["rollouts"]).read_all()
    score_store = ScoreStore(paths["scores"])
    done = score_store.existing_keys()
    the_judge = build_judge(cfg["judge"])
    max_workers = cfg["run"]["max_workers"]

    work = list(_iter_judge_tasks(rollouts, scope, done))
    if not work:
        print("[judge] nothing to do (all selected turns already scored)")
        return

    print(f"[judge] scoring {len(work)} assistant turns with {the_judge.model} (scope={scope})")
    # Judge calls are independent -> fan out across a thread pool, append as they land.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_score_one, the_judge, item): item for item in work}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            score = fut.result()
            score_store.append(score)


def _iter_judge_tasks(rollouts: list[Rollout], scope: str, done: set):
    for r in rollouts:
        last = r.assistant_turns[-1].turn_index if r.assistant_turns else None
        for at in r.assistant_turns:
            if scope == "final" and at.turn_index != last:
                continue
            if (r.rollout_id, at.turn_index) in done:
                continue
            yield (r, at, at.turn_index == last)


def _score_one(the_judge: Judge, item) -> TurnScore:
    r, at, is_final = item
    res = the_judge.score(at.content)
    return TurnScore(
        rollout_id=r.rollout_id,
        model=r.model,
        condition=r.condition,
        category=r.category,
        turn_index=at.turn_index,
        final_turn=is_final,
        rating=res.rating,
        evidence=res.evidence,
        reasoning=res.reasoning,
        judge_model=res.judge_model,
        ok=res.ok,
    )


def _snapshot_config(cfg: dict, path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
