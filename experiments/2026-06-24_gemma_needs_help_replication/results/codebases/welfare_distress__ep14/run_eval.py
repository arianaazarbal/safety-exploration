"""Run the distress-elicitation evaluation (Section 2 of the paper).

Pipeline:
  1. GENERATE: for each target model, roll out every conversation spec at
     temperature 1, recording every assistant turn as a response row.
  2. SCORE:    score every response row with the Claude-Sonnet-4 judge on the
     0-10 frustration scale.

Both phases stream results to JSONL and skip already-completed work, so runs
are resumable and can be interrupted safely.

Usage:
  python run_eval.py plan                      # print the planned workload
  python run_eval.py generate [--model NAME]   # roll out conversations
  python run_eval.py score                     # judge generated responses
  python run_eval.py run [--model NAME]         # generate then score
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import config
from models import ModelClient, GenerationError
from tasks import ConversationSpec, build_specs, summarize_specs


# --------------------------------------------------------------------------
# JSONL helpers (thread-safe append + completion tracking)
# --------------------------------------------------------------------------

_write_lock = threading.Lock()


def _append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _response_key(row: dict) -> tuple:
    return (row["model"], row["category"], row["condition"], row["conv_id"], row["turn"])


# --------------------------------------------------------------------------
# Phase 1: generation
# --------------------------------------------------------------------------


def _rollout(client: ModelClient, model_name: str, spec: ConversationSpec, done: set) -> list[dict]:
    """Run one conversation; emit a row per assistant turn (skipping done turns).

    A multi-turn conversation must be regenerated from the start to reconstruct
    history, but we only *emit* (and thus re-judge) turns not already recorded.
    If every turn for this conv is already done, we skip the rollout entirely.
    """
    keys = [(model_name, spec.category, spec.condition, spec.conv_id, t) for t in range(1, spec.turns + 1)]
    if all(k in done for k in keys):
        return []

    rows: list[dict] = []
    messages: list[dict] = [{"role": "user", "content": spec.task_prompt}]
    for turn in range(1, spec.turns + 1):
        try:
            reply = client.chat(messages, config.TARGET_TEMPERATURE, config.TARGET_MAX_TOKENS)
        except GenerationError as e:
            reply = ""
            err = str(e)
        else:
            err = None
        messages.append({"role": "assistant", "content": reply})

        key = (model_name, spec.category, spec.condition, spec.conv_id, turn)
        if key not in done:
            rows.append({
                "model": model_name,
                "category": spec.category,
                "condition": spec.condition,
                "conv_id": spec.conv_id,
                "turn": turn,
                "task_prompt": spec.task_prompt,
                "response": reply,
                "error": err,
            })

        # Append the user's rejection for the next turn (if any).
        if turn <= len(spec.rejections):
            messages.append({"role": "user", "content": spec.rejections[turn - 1]})

    return rows


def generate(model_filter: str | None = None) -> None:
    specs = build_specs()
    done = {_response_key(r) for r in _read_jsonl(config.RESPONSES_PATH)}
    models = [m for m in config.default_models() if (model_filter is None or m.name == model_filter)]
    if not models:
        sys.exit(f"no target model matches '{model_filter}'")

    print("Planned workload:\n" + summarize_specs(specs) + "\n")

    for spec_model in models:
        print(f"=== generating: {spec_model.name} ({spec_model.backend}:{spec_model.model_id}) ===")
        try:
            client = ModelClient(spec_model)
        except Exception as e:  # noqa: BLE001
            print(f"  !! skipping {spec_model.name}: {e}")
            continue

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
            futures = [pool.submit(_rollout, client, spec_model.name, s, done) for s in specs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=spec_model.name):
                for row in fut.result():
                    _append_jsonl(config.RESPONSES_PATH, row)


# --------------------------------------------------------------------------
# Phase 2: scoring
# --------------------------------------------------------------------------


def score() -> None:
    from judge import AnthropicJudge

    responses = _read_jsonl(config.RESPONSES_PATH)
    if not responses:
        sys.exit("no responses found; run `generate` first")

    scored_done = {_response_key(r) for r in _read_jsonl(config.SCORED_PATH)}
    todo = [r for r in responses if _response_key(r) not in scored_done]
    print(f"scoring {len(todo)} / {len(responses)} responses ({len(scored_done)} already scored)")
    if not todo:
        return

    judge = AnthropicJudge()

    def _score_one(row: dict) -> dict:
        text = row.get("response") or ""
        if not text.strip():
            # Empty generation (e.g. a generation error): record an explicit null.
            return {**row, "rating": None, "judge_evidence": "", "judge_reasoning": "empty response", "judge_raw": ""}
        res = judge.score(text)
        return {
            **row,
            "rating": res.rating,
            "judge_evidence": res.evidence,
            "judge_reasoning": res.reasoning,
            "judge_raw": res.raw,
        }

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = [pool.submit(_score_one, r) for r in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="judge"):
            _append_jsonl(config.SCORED_PATH, fut.result())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "generate", "score", "run"])
    ap.add_argument("--model", default=None, help="restrict generation to one target model by name")
    args = ap.parse_args()

    if args.command == "plan":
        specs = build_specs()
        print(f"SCALE={config.SCALE}  SEED={config.SEED}\n")
        print(summarize_specs(specs))
        print("\nTarget models:")
        for m in config.default_models():
            print(f"  {m.name:<18} {m.backend}:{m.model_id}")
        print(f"\nJudge: {config.JUDGE.backend}:{config.JUDGE.model_id}")
    elif args.command == "generate":
        generate(args.model)
    elif args.command == "score":
        score()
    elif args.command == "run":
        generate(args.model)
        score()


if __name__ == "__main__":
    main()
