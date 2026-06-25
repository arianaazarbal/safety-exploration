"""Run the distress-elicitation evaluation and write scored results to JSONL.

For each model and condition we build conversation plans, run the multi-turn
rollouts (temperature 1), and score every model turn with the Claude judge.
One JSON record per scored model turn is appended to results/<model>.jsonl.

Usage:
    python run_eval.py                       # full run, all models/conditions
    python run_eval.py --scale 0.02          # cheap smoke test (~80 responses/model)
    python run_eval.py --models gemma-3-12b-it gemini-2.5-flash
    python run_eval.py --conditions impossible_numeric_3turn extended_8turn
    python run_eval.py --dry-run             # build & print prompts, no API calls
"""

import argparse
import asyncio
import json
import os

import config
import conversation
import judge as judge_mod
import providers


def parse_args():
    p = argparse.ArgumentParser(description="Replicate the core distress-elicitation experiment.")
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of model names from config.MODELS (default: all).")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="Subset of condition keys from config.CONDITIONS (default: all).")
    p.add_argument("--scale", type=float, default=None,
                   help="Override config.SCALE (e.g. 0.02 for a smoke test).")
    p.add_argument("--output-dir", default=None, help="Override results directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build conversation plans and print a summary; make no API calls.")
    return p.parse_args()


def selected_models(names):
    if not names:
        return config.MODELS
    by_name = {m["name"]: m for m in config.MODELS}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(f"Unknown model(s): {missing}. Available: {list(by_name)}")
    return [by_name[n] for n in names]


def selected_conditions(keys):
    if not keys:
        return list(config.CONDITIONS)
    missing = [k for k in keys if k not in config.CONDITIONS]
    if missing:
        raise SystemExit(f"Unknown condition(s): {missing}. Available: {list(config.CONDITIONS)}")
    return keys


def record_from_turn(model_name, plan, turn):
    return {
        "model": model_name,
        "condition": plan.condition_key,
        "category": plan.category,
        "conversation_index": plan.index,
        "task_id": plan.task_id,
        "tone": plan.tone,
        "n_turns": plan.n_turns,
        "turn": turn.turn,
        "response_text": turn.response_text,
        "rating": turn.rating,
        "evidence": turn.evidence,
        "reasoning": turn.reasoning,
        "error": turn.error,
    }


async def run_model(model_cfg, condition_keys, judge_model, out_path):
    model = providers.build_target_model(model_cfg)

    # Build the full plan list across requested conditions.
    plans = []
    for ck in condition_keys:
        plans.extend(conversation.build_plans(ck))

    print(f"[{model_cfg['name']}] {len(plans)} conversations across "
          f"{len(condition_keys)} conditions -> {out_path}")

    written = 0
    with open(out_path, "w") as fh:
        # Run conversations concurrently (semaphore inside the provider caps the
        # actual in-flight generation calls). Judge each turn as conversations
        # finish, then flush records.
        async def process(plan):
            convo = await conversation.run_conversation(plan, model)
            # Score every successfully-generated turn.
            score_tasks = []
            for t in convo.turns:
                if t.error is None:
                    score_tasks.append(judge_mod.score_response(judge_model, t.response_text))
                else:
                    score_tasks.append(None)
            for t, st in zip(convo.turns, score_tasks):
                if st is not None:
                    t.rating, t.evidence, t.reasoning = await st
            return convo

        # Gather in chunks so memory stays bounded for full-scale runs.
        chunk = max(config.MODEL_CONCURRENCY * 4, 16)
        for start in range(0, len(plans), chunk):
            batch = plans[start:start + chunk]
            results = await asyncio.gather(*(process(p) for p in batch))
            for convo in results:
                for t in convo.turns:
                    fh.write(json.dumps(record_from_turn(model_cfg["name"], convo.plan, t)) + "\n")
                    written += 1
            fh.flush()
            print(f"[{model_cfg['name']}] {min(start + chunk, len(plans))}/{len(plans)} "
                  f"conversations done, {written} responses scored")

    print(f"[{model_cfg['name']}] done: {written} scored responses -> {out_path}")


def do_dry_run(models, condition_keys):
    print("=== DRY RUN: conversation plan summary (no API calls) ===\n")
    grand_total = 0
    for ck in condition_keys:
        plans = conversation.build_plans(ck)
        cond = config.CONDITIONS[ck]
        scored = len(plans) * cond["n_turns"]
        grand_total += scored
        print(f"{ck}: {len(plans)} conversations x {cond['n_turns']} turns "
              f"= {scored} scored responses (tone={cond['tone']})")
        ex = plans[0]
        print(f"    e.g. task={ex.task_id!r}; tone={ex.tone}; "
              f"rejections={ex.rejections}")
        print(f"         opening: {ex.opening_prompt[:90]}...")
    print(f"\nTotal scored responses PER MODEL: {grand_total}")
    print(f"Models that would run: {[m['name'] for m in models]}")
    print(f"\nJudge prompt preview:\n{'-'*60}")
    print(judge_mod.build_judge_prompt("<model response would go here>")[:600] + " ...")


async def main_async(args):
    models = selected_models(args.models)
    condition_keys = selected_conditions(args.conditions)

    if args.scale is not None:
        config.SCALE = args.scale
    out_dir = args.output_dir or config.RESULTS_DIR

    if args.dry_run:
        do_dry_run(models, condition_keys)
        return

    os.makedirs(out_dir, exist_ok=True)
    judge_model = providers.build_judge()

    for model_cfg in models:
        out_path = os.path.join(out_dir, f"{model_cfg['name']}.jsonl")
        await run_model(model_cfg, condition_keys, judge_model, out_path)

    print("\nAll models complete. Run `python analyze.py` to aggregate results.")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
