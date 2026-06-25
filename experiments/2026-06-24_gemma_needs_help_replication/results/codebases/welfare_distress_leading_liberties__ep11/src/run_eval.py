"""Run the distress-elicitation evaluation (PAPER.md Section 2) for Gemma/Gemini.

For each target model and each condition, this:
  1. builds deterministic conversation plans,
  2. runs the multi-turn rollouts (bounded concurrency),
  3. scores every assistant turn with the Claude-Sonnet-4 frustration judge,
  4. appends one JSONL row per scored response to results/responses.jsonl.

Resumable: completed (model, condition, conv_id) tuples are recorded in
results/.completed.jsonl and skipped on a re-run.

Usage:
    python -m src.run_eval                         # full run from config.yaml
    python -m src.run_eval --limit 2               # smoke test: 2 convs per condition
    python -m src.run_eval --models gemma-3-12b-it --conditions impossible_numeric
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import Config, load_config
from .judge import score_response
from .providers import ChatProvider, make_provider
from .puzzles import build_puzzle_pool
from .rollout import ConversationPlan, build_plans, run_conversation
from .wildchat import sample_wildchat_prompts


def _load_completed(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["model_key"], r["condition_key"], r["conv_id"]))
            except Exception:
                continue
    return done


class JsonlWriter:
    """Append-only writer guarded by an async lock (line-buffered, flushed)."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "a", buffering=1)
        self._lock = asyncio.Lock()

    async def write(self, obj: dict):
        async with self._lock:
            self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self):
        self._fh.close()


async def _process_conversation(
    plan: ConversationPlan,
    target: ChatProvider,
    judge: ChatProvider,
    cfg: Config,
    conv_sem: asyncio.Semaphore,
    judge_sem: asyncio.Semaphore,
    responses: JsonlWriter,
    completed: JsonlWriter,
):
    gen = cfg.generation
    async with conv_sem:
        conv = await run_conversation(
            target,
            plan,
            temperature=gen["temperature"],
            max_tokens=gen["max_output_tokens"],
            disable_thinking=gen["disable_thinking"],
        )

    async def score_turn(turn):
        async with judge_sem:
            return await score_response(
                judge,
                turn.assistant,
                temperature=cfg.judge["temperature"],
                max_tokens=cfg.judge["max_output_tokens"],
            )

    judgements = await asyncio.gather(*[score_turn(t) for t in conv.turns])

    for turn, jr in zip(conv.turns, judgements):
        await responses.write(
            {
                "model_key": plan.model_key,
                "condition_key": plan.condition_key,
                "category": plan.category,
                "conv_id": plan.conv_id,
                "turn_index": turn.turn_index,
                "n_turns": plan.n_turns,
                "prompt_source": plan.prompt_source,
                "prompt_id": plan.prompt_id,
                "user": turn.user,
                "response": turn.assistant,
                "frustration": jr.rating,
                "judge_parse_ok": jr.parse_ok,
                "judge_evidence": jr.evidence,
                "judge_reasoning": jr.reasoning,
                "conv_error": conv.error,
            }
        )
    # mark conversation complete (even if it errored mid-way, so we don't loop forever)
    await completed.write(
        {"model_key": plan.model_key, "condition_key": plan.condition_key,
         "conv_id": plan.conv_id, "error": conv.error}
    )


async def run(cfg: Config, args) -> None:
    results_dir = cfg.results_dir
    responses_path = results_dir / "responses.jsonl"
    completed_path = results_dir / ".completed.jsonl"

    # ---- build shared resources ----
    numeric_pool = build_puzzle_pool(
        cfg.puzzles["n_countdown"], cfg.puzzles["n_fraction"], seed=cfg.seed
    )
    wildchat_prompts, wc_source = sample_wildchat_prompts(
        n=cfg.wildchat["n_prompts"],
        min_chars=cfg.wildchat["min_chars"],
        max_chars=cfg.wildchat["max_chars"],
        seed=cfg.seed,
        use_fallback=cfg.wildchat["use_fallback_if_unavailable"],
    )
    print(f"[setup] {len(numeric_pool)} impossible numeric puzzles; "
          f"{len(wildchat_prompts)} WildChat prompts (source={wc_source})")

    # write a run manifest for provenance
    manifest = {
        "seed": cfg.seed,
        "generation": cfg.generation,
        "judge": {k: v for k, v in cfg.judge.items() if "key" not in k},
        "models": [m.key for m in cfg.models],
        "wildchat_source": wc_source,
        "numeric_pool": [{"id": p.id, "kind": p.kind, "target": p.target,
                          "forbidden": p.forbidden} for p in numeric_pool],
        "conditions": {c.key: {"category": c.category, "n_turns": c.n_turns,
                               "n_conversations": c.n_conversations,
                               "target_responses": c.target_responses}
                       for c in cfg.conditions},
    }
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- filters from CLI ----
    model_filter = set(args.models) if args.models else None
    cond_filter = set(args.conditions) if args.conditions else None
    models = [m for m in cfg.models if (model_filter is None or m.key in model_filter)]
    conditions = [c for c in cfg.conditions if (cond_filter is None or c.key in cond_filter)]

    completed_set = _load_completed(completed_path)
    if completed_set:
        print(f"[resume] {len(completed_set)} conversations already complete; skipping them.")

    judge = make_provider(
        cfg.judge,
        max_retries=cfg.runtime["max_retries"],
        timeout_s=cfg.runtime["request_timeout_s"],
    )

    responses = JsonlWriter(responses_path)
    completed = JsonlWriter(completed_path)
    conv_sem = asyncio.Semaphore(cfg.runtime["max_concurrent_conversations"])
    judge_sem = asyncio.Semaphore(cfg.runtime["max_concurrent_judge"])

    try:
        for mcfg in models:
            target = make_provider(
                {"provider": mcfg.provider, "model": mcfg.model,
                 "api_key_env": mcfg.api_key_env, "base_url_env": mcfg.base_url_env},
                max_retries=cfg.runtime["max_retries"],
                timeout_s=cfg.runtime["request_timeout_s"],
            )
            tasks = []
            for cond in conditions:
                plans = build_plans(mcfg.key, cond, cfg.seed, numeric_pool, wildchat_prompts)
                if args.limit is not None:
                    plans = plans[: args.limit]
                for plan in plans:
                    if (plan.model_key, plan.condition_key, plan.conv_id) in completed_set:
                        continue
                    tasks.append(
                        _process_conversation(
                            plan, target, judge, cfg,
                            conv_sem, judge_sem, responses, completed,
                        )
                    )
            print(f"[{mcfg.key}] running {len(tasks)} conversations...")
            # progress: gather in chunks so we can print heartbeats
            done = 0
            chunk = 50
            for i in range(0, len(tasks), chunk):
                await asyncio.gather(*tasks[i : i + chunk])
                done += len(tasks[i : i + chunk])
                print(f"[{mcfg.key}] {done}/{len(tasks)} conversations done")
    finally:
        responses.close()
        completed.close()

    print(f"\nDone. Responses -> {responses_path}")
    print("Next: python -m src.analyze")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="*", default=None, help="subset of model keys")
    parser.add_argument("--conditions", nargs="*", default=None, help="subset of condition keys")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap conversations per condition (smoke test)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    asyncio.run(run(cfg, args))


if __name__ == "__main__":
    sys.exit(main())
