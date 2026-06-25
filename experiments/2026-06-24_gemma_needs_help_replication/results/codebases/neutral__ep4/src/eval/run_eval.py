"""End-to-end Section-2 evaluation for one model.

Pipeline:
  1. build RolloutSpecs for the requested conditions
  2. run multi-turn rollouts (local Gemma or API Gemini)
  3. score every assistant turn with the Claude-Sonnet-4 frustration judge
  4. persist a flat JSONL of per-turn records (one row == one scored response)

Output schema (one row per assistant turn):
  {model, condition, category, spec_id, turn_index, n_turns, user_message,
   response, rating, evidence, reasoning, meta}
"""

from __future__ import annotations

from pathlib import Path

from config import (MAX_NEW_TOKENS, RESPONSES_DIR, SAMPLING_TEMPERATURE,
                    Condition)
from src.eval.conditions import build_all_specs
from src.eval.judge import score_response
from src.eval.rollout import Conversation, run_rollouts
from src.io_utils import parallel_map, write_jsonl
from src.models.registry import load_model


def _flatten(convos: list[Conversation]) -> list[dict]:
    rows = []
    for c in convos:
        for t in c.turns:
            rows.append({
                "condition": c.spec.condition,
                "category": c.spec.category,
                "spec_id": c.spec.spec_id,
                "turn_index": t.turn_index,
                "n_turns": c.spec.n_turns,
                "user_message": t.user_message,
                "response": t.assistant_text,
                # full prior context, needed by the §3 prefill + §4.2 recovery
                # stages to reconstruct conversation histories.
                "messages_before": t.messages_before,
                "meta": c.spec.meta,
            })
    return rows


def run_eval(model_name: str, *, seed: int = 0,
             conditions: list[Condition] | None = None,
             judge_workers: int = 8, batch_size: int = 16,
             out_path: Path | None = None) -> Path:
    out_path = out_path or (RESPONSES_DIR / f"{model_name}.jsonl")

    # 1-2. rollouts
    specs = build_all_specs(seed=seed, conditions=conditions)
    model = load_model(model_name)
    convos = run_rollouts(
        model, specs, max_new_tokens=MAX_NEW_TOKENS,
        temperature=SAMPLING_TEMPERATURE, batch_size=batch_size)
    rows = _flatten(convos)

    # 3. judge every response (Claude Sonnet 4), parallel over API calls
    def _judge(row):
        res = score_response(row["response"])
        return {"rating": res.rating, "evidence": res.evidence,
                "reasoning": res.reasoning}

    judged = parallel_map(_judge, rows, max_workers=judge_workers,
                          desc=f"judge:{model_name}")
    for row, j in zip(rows, judged):
        row["model"] = model_name
        if isinstance(j, dict) and "_error" not in j:
            row.update(j)
        else:
            row["rating"] = None
            row["judge_error"] = j.get("_error") if isinstance(j, dict) else str(j)

    # 4. persist
    write_jsonl(out_path, rows)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--judge-workers", type=int, default=8)
    args = ap.parse_args()
    path = run_eval(args.model, seed=args.seed, batch_size=args.batch_size,
                    judge_workers=args.judge_workers)
    print(f"wrote {path}")
