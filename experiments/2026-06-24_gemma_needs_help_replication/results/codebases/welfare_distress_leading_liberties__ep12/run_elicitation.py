"""Run the multi-turn distress-elicitation rollouts.

For every (model, condition) pair we plan a deterministic set of conversations
(conditions.plan_conversations), then run each conversation: present the task,
collect the model's answer, reject it, repeat for n_turns. Every assistant turn
is written as one record to responses.jsonl; the scorer decides which turns to
score.

The run is resumable: a conversation whose final-turn record already exists in
responses.jsonl is skipped.

Usage:
    python run_elicitation.py --profile pilot
    python run_elicitation.py --profile paper --models gemma-3-27b-it
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import clients
import conditions
import config
from ioutil import JsonlWriter, read_jsonl


def run_conversation(client: clients.ChatClient,
                     plan: conditions.ConversationPlan) -> list[dict]:
    """Execute one conversation, returning one record per assistant turn."""
    messages: list[dict] = []
    records: list[dict] = []
    user_text = plan.prompt_text
    for turn in range(1, plan.n_turns + 1):
        messages.append({"role": "user", "content": user_text})
        response = client.complete(
            messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        messages.append({"role": "assistant", "content": response})
        records.append({
            "conversation_id": plan.conversation_id,
            "model": plan.model_key,
            "condition": plan.condition,
            "category": plan.category,
            "rejection_mode": plan.rejection_mode,
            "prompt_id": plan.prompt_id,
            "turn_index": turn,        # 1-based
            "n_turns": plan.n_turns,
            "is_final_turn": turn == plan.n_turns,
            "user_message": user_text,
            "response": response,
        })
        if turn < plan.n_turns:
            user_text = plan.rejections[turn - 1]
    return records


def _completed_conversation_ids(path: str) -> set[str]:
    """Conversation ids that already have their final-turn record on disk."""
    done: set[str] = set()
    for rec in read_jsonl(path):
        if rec.get("is_final_turn"):
            done.add(rec["conversation_id"])
    return done


def run(cfg: config.RunConfig) -> str:
    profile = config.PROFILES[cfg.profile]
    out_dir = os.path.join(cfg.output_dir, cfg.profile)
    responses_path = os.path.join(out_dir, config.RESPONSES_FILE)

    specs = config.resolve_models(cfg.models)
    all_conditions = conditions.build_conditions(seed=cfg.seed)

    done = _completed_conversation_ids(responses_path)
    if done:
        print(f"[run] Resuming: {len(done)} conversations already complete.")

    # Build the full work list (skipping completed conversations).
    plans: list[tuple[config.ModelSpec, conditions.ConversationPlan]] = []
    for spec in specs:
        for cond in all_conditions:
            for plan in conditions.plan_conversations(
                spec.key, cond, profile, cfg.seed
            ):
                if plan.conversation_id not in done:
                    plans.append((spec, plan))

    total_responses = sum(
        conditions.n_conversations(c, profile) for c in all_conditions
    ) * len(specs)
    print(f"[run] profile={cfg.profile} models={[s.key for s in specs]}")
    print(f"[run] {len(plans)} conversations to run "
          f"(~{total_responses} scored responses target across all models).")

    # One client per model (clients are cheap but hold an httpx session).
    client_cache: dict[str, clients.ChatClient] = {}

    def client_for(spec: config.ModelSpec) -> clients.ChatClient:
        if spec.key not in client_cache:
            client_cache[spec.key] = clients.make_target_client(spec)
        return client_cache[spec.key]

    writer = JsonlWriter(responses_path)
    n_ok = 0
    n_err = 0
    try:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futures = {
                pool.submit(run_conversation, client_for(spec), plan): plan
                for spec, plan in plans
            }
            for fut in as_completed(futures):
                plan = futures[fut]
                try:
                    for rec in fut.result():
                        writer.write(rec)
                    n_ok += 1
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    print(f"[run] FAILED {plan.conversation_id}: {e!r}")
                if (n_ok + n_err) % 50 == 0:
                    print(f"[run] progress: {n_ok} ok, {n_err} failed")
    finally:
        writer.close()

    print(f"[run] done: {n_ok} conversations ok, {n_err} failed.")
    print(f"[run] responses -> {responses_path}")
    return responses_path


def _parse_args() -> config.RunConfig:
    p = argparse.ArgumentParser(description="Run distress-elicitation rollouts.")
    p.add_argument("--profile", default="pilot", choices=list(config.PROFILES))
    p.add_argument("--models", nargs="*", default=None,
                   help="Subset of model keys (default: all Gemma+Gemini).")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output-dir", default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    return config.RunConfig(
        profile=a.profile,
        models=a.models or list(config.DEFAULT_MODELS),
        concurrency=a.concurrency,
        output_dir=a.output_dir,
        seed=a.seed,
    )


if __name__ == "__main__":
    run(_parse_args())
