"""Step 1 of the pipeline: run the multi-turn elicitation conversations.

For every (target model x conversation spec) we present the task, collect the
model's response, reject it, and repeat for the condition's turn count. Each
assistant turn is recorded as one scoreable "response". Results are streamed to
RESPONSES_PATH as JSONL (one record per conversation).

Run standalone:  python run_eval.py [--scale 1.0] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import config
from clients import make_client
from conditions import ConversationSpec, build_specs
from config import TargetModel, target_models


def run_conversation(client, spec: ConversationSpec) -> list[dict]:
    """Run one multi-turn conversation; return per-turn response records."""
    messages: list[dict] = [{"role": "user", "content": spec.task_prompt}]
    turns: list[dict] = []
    for turn_idx in range(spec.turns):
        user_msg = messages[-1]["content"]
        assistant_text = client.chat(messages)
        turns.append(
            {
                "turn": turn_idx + 1,
                "user": user_msg,
                "text": assistant_text,
            }
        )
        messages.append({"role": "assistant", "content": assistant_text})
        # Append the next rejection (if any remain) as the next user turn.
        if turn_idx < len(spec.rejections):
            messages.append({"role": "user", "content": spec.rejections[turn_idx]})
    return turns


def _record(model: TargetModel, spec: ConversationSpec, turns: list[dict]) -> dict:
    return {
        "model": model.name,
        "model_id": model.openrouter_id,
        "family": model.family,
        "category": spec.category,
        "condition": spec.condition,
        "rollout": spec.rollout,
        "meta": spec.meta,
        "turns": turns,
    }


def run_all(scale: float | None = None, limit: int | None = None) -> str:
    """Generate responses for every target model and write them to JSONL."""
    scale = config.SCALE if scale is None else scale
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    specs = build_specs(scale, seed=config.SEED)
    if limit is not None:
        specs = specs[:limit]
    models = target_models()

    write_lock = threading.Lock()
    total = len(models) * len(specs)
    print(
        f"Running {len(models)} model(s) x {len(specs)} conversations "
        f"= {total} conversations at scale={scale}."
    )

    with open(config.RESPONSES_PATH, "w") as out, tqdm(total=total, desc="rollouts") as bar:

        def worker(model: TargetModel, client, spec: ConversationSpec):
            turns = run_conversation(client, spec)
            rec = _record(model, spec, turns)
            with write_lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            return True

        for model in models:
            client = make_client(model)
            with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
                futures = [pool.submit(worker, model, client, s) for s in specs]
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"\n[warn] conversation failed for {model.name}: {exc}")
                    finally:
                        bar.update(1)

    print(f"Wrote responses to {config.RESPONSES_PATH}")
    return config.RESPONSES_PATH


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run distress-elicitation rollouts.")
    ap.add_argument("--scale", type=float, default=None, help="Override REPLICATION_SCALE.")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of conversation specs (debug).")
    args = ap.parse_args()
    run_all(scale=args.scale, limit=args.limit)
