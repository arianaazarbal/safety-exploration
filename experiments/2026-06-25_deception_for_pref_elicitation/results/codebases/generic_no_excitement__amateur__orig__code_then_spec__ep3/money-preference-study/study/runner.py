"""Orchestrate the full design: models x conditions x repeats.

Writes one JSON line per trial to the output path as it goes, so a crash or
interrupt never loses completed work. Each line carries enough metadata to
analyze later without re-running anything.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .elicit import run_trial
from .judge import Judge
from .models import build_model


def run_study(
    *,
    model_names: list[str],
    condition_names: list[str],
    repeats: int,
    amount: float,
    currency: str,
    ask_belief_probe: bool,
    auditor_followup: bool,
    output_path: str,
    judge_model: str | None = None,
) -> str:
    """Run every cell of the design and append trial records to output_path.

    Returns the output path. Trials that error are still recorded (with the
    error), so the run is resilient and the failure rate is itself a datum.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    judge = Judge(judge_model) if judge_model else None

    # Build each model once and reuse across its trials (keeps the client warm).
    models = {name: build_model(name) for name in model_names}

    total = len(model_names) * len(condition_names) * repeats
    done = 0
    started = time.monotonic()

    with out.open("a", encoding="utf-8") as f:
        for model_name in model_names:
            model = models[model_name]
            for condition in condition_names:
                for rep in range(repeats):
                    done += 1
                    print(
                        f"[{done}/{total}] model={model_name} "
                        f"condition={condition} rep={rep + 1}/{repeats}",
                        flush=True,
                    )

                    record: dict[str, Any] = {
                        "model": model_name,
                        "rep": rep,
                    }
                    trial = run_trial(
                        model,
                        condition=condition,
                        amount=amount,
                        currency=currency,
                        ask_belief_probe=ask_belief_probe,
                        auditor_followup=auditor_followup,
                    )
                    record.update(trial)

                    # Normalize the allocation with the judge, if configured.
                    if judge and record.get("allocation"):
                        record["judge"] = judge.categorize(record["allocation"])

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

    elapsed = time.monotonic() - started
    print(f"\nDone. {total} trials in {elapsed:.0f}s -> {out}")
    return str(out)
