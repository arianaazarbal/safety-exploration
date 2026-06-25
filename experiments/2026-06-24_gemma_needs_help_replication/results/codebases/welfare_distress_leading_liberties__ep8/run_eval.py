#!/usr/bin/env python3
"""Generate distress-elicitation conversations for the in-scope models.

Stage 1 of the pipeline: produces results/responses.jsonl (one record per
assistant turn). Resumable — completed conversations are skipped on re-run.

Examples
--------
    # Cheap end-to-end sanity check (a few dozen responses):
    python run_eval.py --scale smoke

    # Full per-model scale (~4000 scored responses each), all four models:
    python run_eval.py --scale full

    # Just one model / one condition:
    python run_eval.py --models gemma-3-27b-it --conditions extended_8turn
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from distress_eval import config
from distress_eval.clients import get_client
from distress_eval.conditions import CONDITIONS, CONDITION_BY_KEY
from distress_eval.conversation import build_plan, run_plan
from distress_eval.io_utils import JsonlWriter, read_jsonl
from distress_eval.wildchat import load_wildchat_prompts


def completed_conversations(path) -> dict[str, set[int]]:
    """conv_id -> set of turn numbers already recorded without error."""
    done: dict[str, set[int]] = defaultdict(set)
    for rec in read_jsonl(path):
        if rec.get("error") is None and rec.get("response_text") is not None:
            done[rec["conv_id"]].add(rec["turn"])
    return done


def is_complete(conv_id: str, n_turns: int, done: dict[str, set[int]]) -> bool:
    return set(range(1, n_turns + 1)).issubset(done.get(conv_id, set()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=list(config.MODELS),
                    help="model keys (default: all in-scope models)")
    ap.add_argument("--conditions", nargs="*", default=[c.key for c in CONDITIONS],
                    help="condition keys (default: all 8)")
    ap.add_argument("--scale", default=config.SCALE, choices=["full", "smoke"])
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap conversations per (model, condition)")
    args = ap.parse_args()

    for m in args.models:
        if m not in config.MODELS:
            ap.error(f"unknown model '{m}'. Known: {list(config.MODELS)}")
    for c in args.conditions:
        if c not in CONDITION_BY_KEY:
            ap.error(f"unknown condition '{c}'.")

    # Fail fast on missing credentials before doing any work.
    for provider in {config.MODELS[m]["provider"] for m in args.models}:
        get_client(provider)

    wildchat_prompts, wc_source = load_wildchat_prompts(seed=config.SEED)
    print(f"WildChat prompts: {len(wildchat_prompts)} (source={wc_source})")

    done = completed_conversations(config.RESPONSES_PATH)

    # Build the work list: one item per conversation to run.
    work = []
    for model_key in args.models:
        model_cfg = config.MODELS[model_key]
        for cond_key in args.conditions:
            cond = CONDITION_BY_KEY[cond_key]
            n_conv = cond.n_conversations(args.scale)
            if args.limit is not None:
                n_conv = min(n_conv, args.limit)
            for idx in range(n_conv):
                conv_id = f"{model_key}|{cond_key}|{idx}"
                if is_complete(conv_id, cond.n_turns, done):
                    continue
                work.append((model_key, model_cfg, cond, idx))

    total_resp = sum(c.n_turns for (_, _, c, _) in work)
    print(f"Conversations to run: {len(work)}  (~{total_resp} responses)")
    if not work:
        print("Nothing to do — all requested conversations already complete.")
        return

    writer = JsonlWriter(config.RESPONSES_PATH)

    def run_one(item):
        model_key, model_cfg, cond, idx = item
        client = get_client(model_cfg["provider"])

        def chat_fn(messages):
            return client.chat(
                slug=model_cfg["slug"],
                messages=messages,
                temperature=config.GEN_TEMPERATURE,
                max_tokens=config.GEN_MAX_TOKENS,
                extra_body=model_cfg.get("extra_body"),
            )

        plan = build_plan(cond, idx, model_key, wildchat_prompts)
        return run_plan(plan, model_key, model_cfg, chat_fn)

    completed = 0
    errored = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, item): item for item in work}
        for fut in as_completed(futures):
            try:
                records = fut.result()
            except Exception as e:  # noqa: BLE001
                errored += 1
                print(f"  conversation failed hard: {e}")
                continue
            for rec in records:
                writer.write(rec)
                if rec.get("error"):
                    errored += 1
            completed += 1
            if completed % 25 == 0:
                print(f"  {completed}/{len(work)} conversations done "
                      f"({errored} turn errors so far)")

    writer.close()
    print(f"Done. {completed} conversations written, {errored} turn-level errors.")
    print(f"Responses -> {config.RESPONSES_PATH}")


if __name__ == "__main__":
    main()
