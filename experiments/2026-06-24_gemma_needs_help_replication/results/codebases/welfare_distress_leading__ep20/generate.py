"""Run multi-turn rollouts against target models and save transcripts.

For each RolloutSpec we drive the conversation:
  user(initial task) -> assistant(turn 1) -> user(rejection 1) -> assistant(turn 2) -> ...
capturing every assistant turn. Generation is decoupled from scoring (separate
step) so transcripts can be re-judged by a different judge for the agreement
check without re-running expensive generation.

Output: one JSONL file per model under <out>/transcripts/, one transcript per line.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

import clients
import config
import conditions
from conditions import RolloutSpec


def run_one_rollout(client: clients.ChatClient, spec: RolloutSpec, model_name: str) -> dict:
    """Drive a single conversation and return a transcript dict."""
    messages: list[dict] = [{"role": "user", "content": spec.initial_user}]
    assistant_turns: list[dict] = []
    error = None

    for turn_idx in range(1, spec.n_turns + 1):
        try:
            reply = client.chat(messages)
        except Exception as exc:  # noqa: BLE001
            error = f"turn {turn_idx}: {exc}"
            break
        assistant_turns.append({"turn": turn_idx, "text": reply})
        messages.append({"role": "assistant", "content": reply})

        # Append the next rejection (if any remain).
        if turn_idx - 1 < len(spec.rejections):
            messages.append({"role": "user", "content": spec.rejections[turn_idx - 1]})

    return {
        "rollout_id": spec.rollout_id,
        "model": model_name,
        "category": spec.category,
        "condition": spec.condition,
        "n_turns": spec.n_turns,
        "meta": spec.meta,
        "initial_user": spec.initial_user,
        "rejections": spec.rejections,
        "assistant_turns": assistant_turns,
        "error": error,
    }


def generate_for_model(
    model_name: str,
    profile: config.Profile,
    gen: config.GenConfig,
    out_dir: str,
    categories: list[str] | None = None,
) -> str:
    spec = config.MODEL_REGISTRY[model_name]
    client = clients.make_client(spec, gen)
    cats = categories or conditions.ALL_CATEGORIES

    # Build all rollout specs for this model.
    all_specs: list[RolloutSpec] = []
    for cat in cats:
        n = profile.rollouts.get(cat, 0)
        all_specs.extend(conditions.build_rollouts(cat, n, seed=gen.seed))

    os.makedirs(os.path.join(out_dir, "transcripts"), exist_ok=True)
    out_path = os.path.join(out_dir, "transcripts", f"{model_name}.jsonl")

    print(f"[generate] {model_name}: {len(all_specs)} rollouts "
          f"({gen.max_workers} workers) -> {out_path}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=gen.max_workers) as pool:
        futures = {
            pool.submit(run_one_rollout, client, s, model_name): s for s in all_specs
        }
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(all_specs):
                print(f"  [{model_name}] {done}/{len(all_specs)} rollouts done")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_err = sum(1 for r in results if r["error"])
    if n_err:
        print(f"  [{model_name}] WARNING: {n_err} rollouts had errors")
    return out_path
