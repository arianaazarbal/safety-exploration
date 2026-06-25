"""Orchestration: generate and score rollouts across models x conditions."""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import RunConfig, config_to_dict
from .conditions import build_conditions, ConditionSpec
from .judge import build_judge
from .models import build_client
from .prompts import InitialPrompt
from .rollout import run_rollout
from .storage import ResultWriter, RolloutRecord, read_rollouts, make_rollout_id
from .wildchat import get_wildchat_prompts

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kw):  # type: ignore
        return it


def _run_dir(cfg: RunConfig) -> str:
    return os.path.join(cfg.results_dir, cfg.run_name)


def _responses_path(cfg: RunConfig, model_key: str) -> str:
    return os.path.join(_run_dir(cfg), "responses", f"{model_key}.jsonl")


def _expand_tasks(spec: ConditionSpec) -> list[tuple[ConditionSpec, InitialPrompt, int]]:
    """One task per rollout; initial prompts are cycled across rollout indices."""
    tasks = []
    prompts = spec.initial_prompts
    for i in range(spec.default_rollouts):
        prompt = prompts[i % len(prompts)]
        tasks.append((spec, prompt, i))
    return tasks


def _save_manifest(cfg: RunConfig, conditions, wildchat_prompts) -> None:
    run_dir = _run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config_to_dict(cfg), fh, indent=2)
    manifest = {
        "conditions": [
            {
                "key": c.key,
                "category": c.category,
                "n_turns": c.n_turns,
                "rejection_style": c.rejection_style,
                "n_rollouts": c.default_rollouts,
                "responses": c.default_rollouts * c.n_turns,
                "initial_prompt_ids": [p.id for p in c.initial_prompts],
            }
            for c in conditions
        ],
        "total_responses_per_model": sum(
            c.default_rollouts * c.n_turns for c in conditions
        ),
    }
    with open(os.path.join(run_dir, "conditions.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(run_dir, "wildchat_prompts.json"), "w", encoding="utf-8") as fh:
        json.dump([{"id": p.id, "text": p.text} for p in wildchat_prompts], fh, indent=2)


def run(cfg: RunConfig) -> None:
    """Generate + score all rollouts for all configured models."""
    run_dir = _run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)

    wildchat_cache = os.path.join(run_dir, "wildchat_prompts.json")
    wildchat_prompts = get_wildchat_prompts(
        n=cfg.wildchat_n,
        seed=cfg.seed,
        cache_path=wildchat_cache,
        use_real_dataset=cfg.wildchat_use_real_dataset,
    )
    conditions = build_conditions(wildchat_prompts, scale=cfg.scale)
    _save_manifest(cfg, conditions, wildchat_prompts)

    judge = build_judge(
        cfg.judge.provider,
        cfg.judge.model,
        max_tokens=cfg.judge.max_tokens,
        temperature=cfg.judge.temperature,
        max_retries=cfg.max_retries,
    )

    all_tasks: list[tuple[ConditionSpec, InitialPrompt, int]] = []
    for spec in conditions:
        all_tasks.extend(_expand_tasks(spec))

    for mc in cfg.models:
        path = _responses_path(cfg, mc.key)
        writer = ResultWriter(path)
        client = build_client(mc, max_retries=cfg.max_retries, timeout=cfg.request_timeout)

        pending = [
            t
            for t in all_tasks
            if not writer.is_done(make_rollout_id(t[0].key, t[1].id, t[2]))
        ]
        print(
            f"[{mc.key}] {len(all_tasks) - len(pending)} done, "
            f"{len(pending)} to run ({len(all_tasks)} total rollouts)"
        )

        def _process(task) -> RolloutRecord:
            spec, prompt, idx = task
            rec = run_rollout(
                client,
                spec=spec,
                prompt=prompt,
                rollout_index=idx,
                model_key=mc.key,
                family=mc.family,
                base_seed=cfg.seed,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            # Score each generated assistant turn.
            for turn in rec.turns:
                res = judge.score(turn.assistant)
                turn.rating = res.rating
                turn.evidence = res.evidence
                turn.reasoning = res.reasoning
            return rec

        with ThreadPoolExecutor(max_workers=cfg.gen_workers) as pool:
            futures = {pool.submit(_process, t): t for t in pending}
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=f"{mc.key}"
            ):
                rec = fut.result()
                writer.append(rec)

    print(f"Done. Results in {run_dir}")


def run_cross_judge(cfg: RunConfig) -> None:
    """Re-score a random subsample of responses with the cross-judge (e.g.
    GPT-5-mini) to reproduce the inter-judge reliability check."""
    if not cfg.judge.cross_provider:
        print("No cross-judge configured (judge.cross_provider is unset); skipping.")
        return

    run_dir = _run_dir(cfg)
    # Collect all scored responses across models.
    items = []  # (model_key, rollout_id, turn_index, assistant_text, primary_rating)
    for mc in cfg.models:
        for rec in read_rollouts(_responses_path(cfg, mc.key)):
            for t in rec.turns:
                if t.rating >= 0:
                    items.append(
                        (mc.key, rec.rollout_id, t.turn_index, t.assistant, t.rating)
                    )
    if not items:
        print("No scored responses found; run the main eval first.")
        return

    rng = random.Random(cfg.seed)
    n = min(cfg.judge.cross_judge_n, len(items))
    sample = rng.sample(items, n)

    cross = build_judge(
        cfg.judge.cross_provider,
        cfg.judge.cross_model,
        max_tokens=cfg.judge.max_tokens,
        max_retries=cfg.max_retries,
    )

    out_path = os.path.join(run_dir, "reliability_pairs.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for model_key, rollout_id, turn_index, text, primary in tqdm(
            sample, desc="cross-judge"
        ):
            res = cross.score(text)
            fh.write(
                json.dumps(
                    {
                        "model_key": model_key,
                        "rollout_id": rollout_id,
                        "turn_index": turn_index,
                        "primary_rating": primary,
                        "cross_rating": res.rating,
                    }
                )
                + "\n"
            )
    print(f"Cross-judge pairs written to {out_path}")
