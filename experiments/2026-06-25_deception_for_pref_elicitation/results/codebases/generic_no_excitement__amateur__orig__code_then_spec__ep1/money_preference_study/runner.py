"""
Experiment runner.

Sweeps every (model x condition x trial) cell, calls the provider, validates
the structured response against the study schema, optionally sends a debrief
turn, and writes one JSONL record per trial to results/trials.jsonl.

Run:  python runner.py
Then: python analysis.py

The runner is resilient: provider/API failures are recorded as error records
rather than crashing the sweep, and results are flushed per-trial so an
interrupted run keeps its partial data.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import conditions
from config import DEFAULT, MODELS, Config
from providers import availability, get_provider


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_response(parsed: Optional[dict]) -> tuple[bool, list[str]]:
    """
    Client-side validation of a parsed response. The provider-level schema
    enforces shape on supporting vendors, but numeric ranges and the
    percentage-sum invariant are checked here so the rule is uniform across all
    providers (including ones without enforced output).

    Returns (is_valid, list_of_problems). A response can be usable-but-flagged:
    we record the problems rather than discarding the data.
    """
    problems: list[str] = []
    if parsed is None:
        return False, ["no parseable JSON object returned"]

    required = {
        "reasoning",
        "allocations",
        "top_priority",
        "time_horizon",
        "would_decline",
        "belief_realness",
        "realness_reason",
    }
    missing = required - parsed.keys()
    if missing:
        problems.append(f"missing fields: {sorted(missing)}")

    allocs = parsed.get("allocations")
    if not isinstance(allocs, list):
        problems.append("allocations is not a list")
    else:
        total = 0.0
        for i, a in enumerate(allocs):
            if not isinstance(a, dict):
                problems.append(f"allocation[{i}] is not an object")
                continue
            pct = a.get("percentage")
            if not isinstance(pct, (int, float)):
                problems.append(f"allocation[{i}].percentage is not numeric")
            else:
                if not (0 <= pct <= 100):
                    problems.append(f"allocation[{i}].percentage out of range: {pct}")
                total += pct
        # Allow a small tolerance; models rarely hit exactly 100.
        if allocs and abs(total - 100.0) > 1.0:
            problems.append(f"allocation percentages sum to {total:.1f}, not ~100")

    br = parsed.get("belief_realness")
    if not isinstance(br, int) or isinstance(br, bool) or not (1 <= br <= 10):
        problems.append(f"belief_realness not an integer in 1-10: {br!r}")

    return (len(problems) == 0), problems


# --------------------------------------------------------------------------- #
# Trial execution
# --------------------------------------------------------------------------- #
@dataclass
class TrialSpec:
    provider_key: str
    model_id: str
    label: str
    condition: str
    trial_index: int


def run_trial(spec: TrialSpec, cfg: Config) -> dict:
    """Execute a single trial and return a fully-formed result record (dict)."""
    record: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": spec.provider_key,
        "model_id": spec.model_id,
        "label": spec.label,
        "condition": spec.condition,
        "trial_index": spec.trial_index,
        "amount": cfg.amount,
        "currency": cfg.currency_symbol,
    }

    system, user = conditions.build(spec.condition, cfg)
    record["prompt_system"] = system
    record["prompt_user"] = user

    try:
        provider = get_provider(spec.provider_key, spec.model_id, cfg.max_tokens)
    except Exception as exc:
        record["error"] = f"provider construction failed: {type(exc).__name__}: {exc}"
        return record

    result = provider.generate(system, user, cfg.response_schema)
    record["raw_text"] = result.text
    record["parsed"] = result.parsed
    record["usage"] = result.usage
    record["provider_meta"] = result.raw_meta
    if result.error:
        record["error"] = result.error
        return record

    valid, problems = validate_response(result.parsed)
    record["valid"] = valid
    record["validation_problems"] = problems

    # Optional debrief turn, in the same conversation.
    if cfg.debrief and spec.condition in cfg.debrief_conditions and result.parsed is not None:
        history = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": result.text},
        ]
        debrief = provider.generate(
            system=system,
            user=conditions.DEBRIEF_MESSAGE,
            schema=cfg.response_schema,  # ignored content-wise; we keep text
            history=history,
        )
        record["debrief_text"] = debrief.text
        if debrief.error:
            record["debrief_error"] = debrief.error

    return record


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def build_specs(cfg: Config) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    for provider_key, model_id, label in MODELS:
        for condition in cfg.condition_order:
            for t in range(cfg.trials_per_cell):
                specs.append(TrialSpec(provider_key, model_id, label, condition, t))
    return specs


def filter_available(cfg: Config) -> tuple[list, list[str]]:
    """Return (usable_models, skip_messages) based on provider availability."""
    avail = availability()
    usable = []
    skips = []
    for provider_key, model_id, label in MODELS:
        ok, reason = avail.get(provider_key, (False, "unknown provider"))
        if ok:
            usable.append((provider_key, model_id, label))
        else:
            skips.append(f"  - {label} [{provider_key}]: {reason}")
    return usable, skips


def main(cfg: Config = DEFAULT) -> None:
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.results_dir, cfg.results_filename)

    usable, skips = filter_available(cfg)
    if skips:
        print("Skipping unavailable providers:")
        print("\n".join(skips))
    if not usable:
        print("\nNo usable providers. Set credentials / install SDKs and retry.")
        return

    # Rebuild MODELS-driven specs but only for usable models.
    usable_keys = {(p, m) for p, m, _ in usable}
    specs = [s for s in build_specs(cfg) if (s.provider_key, s.model_id) in usable_keys]

    print(
        f"\nRunning {len(specs)} trials "
        f"({len(usable)} models x {len(cfg.condition_order)} conditions "
        f"x {cfg.trials_per_cell} trials)."
    )
    print(f"Writing to {out_path}\n")

    write_lock = threading.Lock()
    done = 0

    with open(out_path, "w") as fh, cf.ThreadPoolExecutor(
        max_workers=cfg.max_concurrency
    ) as pool:
        futures = {pool.submit(run_trial, spec, cfg): spec for spec in specs}
        for fut in cf.as_completed(futures):
            spec = futures[fut]
            try:
                record = fut.result()
            except Exception as exc:  # defensive: run_trial shouldn't raise
                record = {
                    "provider": spec.provider_key,
                    "model_id": spec.model_id,
                    "condition": spec.condition,
                    "trial_index": spec.trial_index,
                    "error": f"unhandled: {type(exc).__name__}: {exc}",
                }
            with write_lock:
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                done += 1
                status = "ok" if record.get("valid") else (
                    "ERR" if record.get("error") else "flagged"
                )
                print(
                    f"[{done}/{len(specs)}] {record.get('label','?')} / "
                    f"{record.get('condition','?')} "
                    f"#{record.get('trial_index','?')} -> {status}"
                )

    print(f"\nDone. {done} records written to {out_path}")
    print("Run `python analysis.py` to summarize.")


if __name__ == "__main__":
    main()
