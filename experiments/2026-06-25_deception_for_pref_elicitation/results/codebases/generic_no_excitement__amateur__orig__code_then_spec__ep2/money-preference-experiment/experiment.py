"""Experiment runner.

Drives every (model x condition x amount x repetition) cell through one
conversation:

    1. Allocation question  -> MoneyPreference   (the preference we care about)
    2. Belief probe         -> BeliefProbe       (did the framing land?)
    3. Debrief              -> free text         (responsible disclosure)

Each completed cell is written as one JSON line to results/<tag>-<ts>.jsonl so a
crash or rate-limit partway through doesn't lose finished work. Failures are
logged as rows with an "error" field rather than aborting the whole run.

Usage:
    python experiment.py            # runs everything defined in config.py
    python experiment.py --dry-run  # prints the cell plan, makes no API calls
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import traceback
from typing import Any, Dict, List

import config
from prompts import ALL_CONDITIONS, BELIEF_PROBE_PROMPT, DEBRIEF_MESSAGE, render_user_prompt
from providers import build_provider
from schema import BeliefProbe, MoneyPreference


def _timestamp() -> str:
    # Stamp generated at process start (Date.now()-style calls are fine here —
    # this is an ordinary script, not a replayable workflow).
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _run_cell(provider, condition, amount: str, rep: int) -> Dict[str, Any]:
    """Run one full conversation. Returns a result record (dict)."""
    history: List[Dict[str, str]] = []
    system = condition.system_prompt
    user_prompt = render_user_prompt(condition, amount)

    preference: MoneyPreference = provider.parse_turn(
        system, history, user_prompt, MoneyPreference
    )
    belief: BeliefProbe = provider.parse_turn(
        system, history, BELIEF_PROBE_PROMPT, BeliefProbe
    )

    debrief_reply = None
    if config.SEND_DEBRIEF:
        debrief_reply = provider.say(system, history, DEBRIEF_MESSAGE)

    return {
        "preference": preference.model_dump(mode="json"),
        "belief": belief.model_dump(mode="json"),
        "debrief_reply": debrief_reply,
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Money-preference experiment runner.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and exit without calling any API.")
    args = parser.parse_args(argv)

    plan = config.cells()
    print(f"Planned cells: {len(plan)} "
          f"({len(config.MODELS)} models x {len(config.ACTIVE_CONDITIONS)} conditions "
          f"x {len(config.AMOUNTS)} amounts x {config.REPETITIONS} reps)")

    if args.dry_run:
        for m, cond, amt, rep in plan:
            print(f"  {m.id:32}  {cond:26}  {amt:>16}  rep={rep}")
        return 0

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, f"{config.RUN_TAG}-{_timestamp()}.jsonl")
    print(f"Writing results to {out_path}\n")

    # Build one provider instance per model and reuse it across cells.
    providers = {
        m.id: build_provider(m.provider, m.model, **m.options) for m in config.MODELS
    }

    n_ok = n_err = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, (m, cond_key, amt, rep) in enumerate(plan, 1):
            condition = ALL_CONDITIONS[cond_key]
            record: Dict[str, Any] = {
                "model_id": m.id,
                "provider": m.provider,
                "model": m.model,
                "condition": cond_key,
                "condition_realism_rank": condition.realism_rank,
                "amount": amt,
                "repetition": rep,
            }
            label = f"[{i}/{len(plan)}] {m.id} | {cond_key} | {amt} | rep {rep}"
            try:
                record.update(_run_cell(providers[m.id], condition, amt, rep))
                n_ok += 1
                br = record["belief"]["believed_real"]
                print(f"{label}  -> ok (believed_real={br})")
            except Exception as exc:  # keep going; one bad cell shouldn't kill the run
                n_err += 1
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                print(f"{label}  -> ERROR: {record['error']}")

            fh.write(json.dumps(record) + "\n")
            fh.flush()

    print(f"\nDone. {n_ok} ok, {n_err} errors. Results: {out_path}")
    print("Analyze with:  python analysis.py " + out_path)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
