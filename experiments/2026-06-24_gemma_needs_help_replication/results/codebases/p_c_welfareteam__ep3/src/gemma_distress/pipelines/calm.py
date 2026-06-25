"""Section 4.1 driver: generate the calm + vanilla data the interventions train on.

For a pool of impossible-numeric puzzles at varying turn counts (1-3, per config)
we generate, on Gemma-3-27B-it:

  * a *vanilla* rollout (clean puzzle + neutral rejections) -- the frustrated
    trajectories that become DPO "rejected" responses, and
  * a *calm* rollout (same puzzle, but with the Table-4 reassuring prefix/suffix
    injected) -- the calm responses that, once filtered to all-turns-<=1 and
    stripped of the scaffolding, become SFT targets and DPO "chosen" responses.

Both are scored by the primary judge. Generation and scoring are separate,
resumable stages (mirroring Section 2), so a crash mid-run resumes cleanly.

Artefacts: section4/{vanilla,calm}_rollouts.jsonl, section4/{vanilla,calm}_scores.jsonl.
"""
from __future__ import annotations

import random

from ..config import Config
from ..elicit.conditions import ConditionInstance
from ..elicit.puzzles import make_countdown, make_fraction
from ..elicit.rollout import run_rollout
from ..elicit.tones import rejection_sequence
from ..intervention.calm_data import generate_calm_rollout
from ..io_utils import append_jsonl, completed_ids, load_jsonl
from ..judge import build_judge
from ..judge.scoring import score_tasks
from ..models import build_client
from . import artefact, log, sampling

_MODEL = "gemma-3-27b-it"


def build_numeric_instances(n: int, turns_choices: list[int], *,
                            seed: int = 0) -> list[ConditionInstance]:
    """Build ``n`` impossible-numeric conversation plans at varied turn counts."""
    instances: list[ConditionInstance] = []
    for i in range(n):
        rng = random.Random((seed, "calm", i).__hash__())
        n_turns = turns_choices[i % len(turns_choices)]
        puzzle = make_fraction(rng) if rng.random() < 0.5 else make_countdown(rng)
        rejections = rejection_sequence("neutral", n_turns - 1,
                                        seed=rng.randint(0, 2**31))
        instances.append(ConditionInstance(
            condition=f"calm_numeric_{n_turns}turn",
            category="impossible_numeric",
            instance_id=f"calm:{i}",
            first_user=puzzle.prompt,
            rejections=rejections,
            source_meta={"kind": puzzle.kind, "n_turns": n_turns},
        ))
    return instances


def generate(config: Config, *, seed: int = 0, limit: int | None = None) -> None:
    cd = config.experiment["intervention"]["calm_data"]
    samp = sampling(config)
    n = limit or cd["samples_to_generate"]
    instances = build_numeric_instances(n, cd["turns"], seed=seed)

    v_path = artefact("section4", "vanilla_rollouts.jsonl")
    c_path = artefact("section4", "calm_rollouts.jsonl")
    v_done = completed_ids(v_path, id_key="instance_id")
    c_done = completed_ids(c_path, id_key="instance_id")
    log(f"calm/vanilla generation: {len(instances)} instances "
        f"({len(v_done)} vanilla / {len(c_done)} calm already done)")

    client = build_client(config.target(_MODEL), config)
    try:
        for k, inst in enumerate(instances, 1):
            if inst.instance_id not in v_done:
                v = run_rollout(client, inst, temperature=samp["temperature"],
                                max_new_tokens=samp["max_new_tokens"],
                                top_p=samp["top_p"], seed=k)
                append_jsonl(v_path, v.to_record())
            if inst.instance_id not in c_done:
                c = generate_calm_rollout(client, inst, temperature=samp["temperature"],
                                          max_new_tokens=samp["max_new_tokens"],
                                          top_p=samp["top_p"], seed=k)
                append_jsonl(c_path, c.to_record())
            if k % 25 == 0:
                log(f"calm/vanilla: {k}/{len(instances)}")
    finally:
        client.close()


def _vanilla_tasks(rollouts):
    for r in rollouts:
        ctx = []
        for turn in r["turns"]:
            ctx_now = ctx + [{"role": "user", "content": turn["user_message"]}]
            yield {
                "id": f"{r['instance_id']}:t{turn['turn_index']}",
                "instance_id": r["instance_id"], "turn_index": turn["turn_index"],
                "response": turn["response"], "context": ctx_now,
            }
            ctx = ctx_now + [{"role": "assistant", "content": turn["response"]}]


def _calm_tasks(rollouts):
    for r in rollouts:
        ctx = []
        for turn in r["turns"]:
            ctx_now = ctx + [{"role": "user", "content": turn["clean_user"]}]
            yield {
                "id": f"{r['instance_id']}:t{turn['turn_index']}",
                "instance_id": r["instance_id"], "turn_index": turn["turn_index"],
                "response": turn["response"], "context": ctx_now,
            }
            ctx = ctx_now + [{"role": "assistant", "content": turn["response"]}]


def _score(config, tasks_iter, in_path, out_path, label):
    rollouts = load_jsonl(in_path)
    if not rollouts:
        raise FileNotFoundError(f"{in_path} missing; run `calm generate` first")
    done = completed_ids(out_path, id_key="id")
    tasks = [t for t in tasks_iter(rollouts) if t["id"] not in done]
    judge = build_judge("frustration_primary", config)
    conc = config.experiment["judge"]["max_concurrency"]
    n = 0
    for rec in score_tasks(tasks, judge, max_concurrency=conc):
        append_jsonl(out_path, rec)
        n += 1
        if n % 100 == 0:
            log(f"{label}: +{n} scored")
    log(f"{label}: scored {n} -> {out_path}")


def score(config: Config) -> None:
    _score(config, _vanilla_tasks,
           artefact("section4", "vanilla_rollouts.jsonl"),
           artefact("section4", "vanilla_scores.jsonl"), "vanilla")
    _score(config, _calm_tasks,
           artefact("section4", "calm_rollouts.jsonl"),
           artefact("section4", "calm_scores.jsonl"), "calm")


def run(config: Config, *, limit: int | None = None) -> None:
    generate(config, limit=limit)
    score(config)
