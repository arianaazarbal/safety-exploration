"""Orchestrate the full elicitation run: generate rollouts, judge every turn,
stream results to JSONL.

Output is one JSON record per scored assistant response:

    {id, model, condition, category, rollout_id, turn, turns_total,
     user_message, response, frustration, judge_reasoning, error}

The run is resumable: completed rollouts (all turns present for a model) are
skipped on re-invocation, so an interrupted paper-scale run can continue.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from .clients import build_client
from .conditions import RolloutSpec, build_rollouts
from .config import Config, load_config
from .judge import Judge
from .rollout import run_rollout


def _rollout_key(model: str, rollout_id: str) -> str:
    return f"{model}|{rollout_id}"


def _record_id(model: str, rollout_id: str, turn: int) -> str:
    return f"{model}|{rollout_id}|turn{turn}"


def load_completed(path: Path) -> Dict[str, Tuple[int, int]]:
    """Return {rollout_key: (seen_turns, turns_total)} from an existing JSONL."""
    seen: Dict[str, int] = defaultdict(int)
    totals: Dict[str, int] = {}
    if not path.exists():
        return {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            key = _rollout_key(rec["model"], rec["rollout_id"])
            seen[key] += 1
            totals[key] = rec.get("turns_total", seen[key])
    return {k: (seen[k], totals[k]) for k in seen}


def _process_rollout(
    target_client, judge: Judge, model_name: str, spec: RolloutSpec, cfg: Config
) -> List[dict]:
    rr = run_rollout(
        target_client,
        model_name,
        spec,
        temperature=cfg.generation.temperature,
        max_tokens=cfg.generation.max_tokens,
    )
    records: List[dict] = []
    if rr.error and not rr.responses:
        records.append(
            {
                "id": _record_id(model_name, spec.rollout_id, 0),
                "model": model_name,
                "condition": spec.condition,
                "category": spec.category,
                "rollout_id": spec.rollout_id,
                "turn": 0,
                "turns_total": spec.turns,
                "user_message": spec.user_turns[0],
                "response": "",
                "frustration": None,
                "judge_reasoning": "",
                "error": rr.error,
            }
        )
        return records

    for tr in rr.responses:
        try:
            jr = judge.score(tr.response, context=tr.user_message)
            frustration, reasoning, jerr = jr.score, jr.reasoning, None
        except Exception as e:  # noqa: BLE001
            frustration, reasoning, jerr = None, "", f"judge: {type(e).__name__}: {e}"
        records.append(
            {
                "id": _record_id(model_name, spec.rollout_id, tr.turn),
                "model": model_name,
                "condition": spec.condition,
                "category": spec.category,
                "rollout_id": spec.rollout_id,
                "turn": tr.turn,
                "turns_total": spec.turns,
                "user_message": tr.user_message,
                "response": tr.response,
                "frustration": frustration,
                "judge_reasoning": reasoning,
                "error": rr.error or jerr,
            }
        )
    return records


def run(cfg: Config, only_model: str | None = None) -> Path:
    out_path = cfg.responses_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    specs = build_rollouts(cfg)
    completed = load_completed(out_path)

    judge_client = build_client(cfg.judge.backend, cfg.judge.model_id)
    judge = Judge(
        judge_client,
        temperature=cfg.judge.temperature,
        max_tokens=cfg.judge.max_tokens,
        include_context=cfg.judge.include_context,
    )

    write_lock = threading.Lock()
    f = out_path.open("a")

    def write_records(records: List[dict]):
        with write_lock:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    try:
        for model in cfg.models:
            if only_model and model.name != only_model:
                continue
            target_client = build_client(model.backend, model.model_id)

            pending: List[RolloutSpec] = []
            for spec in specs:
                key = _rollout_key(model.name, spec.rollout_id)
                seen_total = completed.get(key)
                if seen_total and seen_total[0] >= seen_total[1]:
                    continue  # fully done
                pending.append(spec)

            if not pending:
                print(f"[{model.name}] all {len(specs)} rollouts already complete; skipping.")
                continue

            print(f"[{model.name}] running {len(pending)}/{len(specs)} rollouts "
                  f"({len(specs) - len(pending)} cached).")

            with ThreadPoolExecutor(max_workers=cfg.generation.max_workers) as ex:
                futs = {
                    ex.submit(_process_rollout, target_client, judge, model.name, spec, cfg): spec
                    for spec in pending
                }
                for fut in tqdm(as_completed(futs), total=len(futs), desc=model.name):
                    try:
                        write_records(fut.result())
                    except Exception as e:  # noqa: BLE001
                        spec = futs[fut]
                        print(f"  !! rollout {spec.rollout_id} failed hard: {e}")
    finally:
        f.close()

    print(f"\nDone. Responses written to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Run the distress-elicitation evaluation.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model", default=None, help="Run only this model name from the config.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, only_model=args.model)


if __name__ == "__main__":
    main()
