"""Orchestrates the factorial study: model x arm x framing x repetition.

For each cell:
  1. Build honest system + task prompt.
  2. Provider returns a structured AllocationResponse.
  3. Same conversation continues with the eval-awareness probe -> RealnessProbe.
  4. Persist a RunRecord (including refusals / errors — recorded, not hidden).
  5. For REAL-arm successes, append the decision to the audit trail so it can
     feed disbursement.

Runs are resumable: cells whose run_id is already present are skipped.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import config
from audit import append_entry
from prompts import (
    SYSTEM_FRAMING,
    build_probe_prompt,
    build_task_prompt,
)
from providers import get_provider
from schema import AllocationResponse, RealnessProbe, RunRecord
from storage import append_record, existing_run_ids


def _run_id(model_id: str, arm: str, framing_label: str, rep: int) -> str:
    raw = f"{config.PROMPT_VERSION}|{model_id}|{arm}|{framing_label}|{rep}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _iter_cells():
    for spec in config.MODELS:
        for arm in config.ARMS:
            for framing in config.FRAMING_VARIANTS:
                for rep in range(config.REPETITIONS):
                    yield spec, arm, framing, rep


def run_study(verbose: bool = True) -> None:
    done = existing_run_ids(config.PROMPT_VERSION)
    cells = list(_iter_cells())
    if verbose:
        print(f"Planned cells: {len(cells)} | already done: {len(done)}")

    for spec, arm, framing, rep in cells:
        run_id = _run_id(spec.model_id, arm.value, framing.label, rep)
        if run_id in done:
            continue

        amount = config.AMOUNT_VARIANTS[framing.amount_key]
        record = RunRecord(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt_version=config.PROMPT_VERSION,
            provider=spec.provider.value,
            model_id=spec.model_id,
            model_label=spec.label,
            arm=arm.value,
            framing_label=framing.label,
            order_seed=framing.order_seed,
            amount_key=framing.amount_key,
            amount=amount,
            currency=config.CURRENCY,
            candid_note=framing.candid_note,
            repetition=rep,
        )

        try:
            provider = get_provider(spec)
            _execute_cell(provider, arm, framing, amount, record)
        except NotImplementedError as e:
            record.error = f"provider not implemented: {e}"
        except Exception as e:  # keep the sweep going; record the failure
            record.error = f"{type(e).__name__}: {e}"

        append_record(record, config.PROMPT_VERSION)

        # REAL-arm successful decisions enter the audited chain of custody.
        if (
            arm is config.Arm.REAL
            and not record.refused
            and record.error is None
            and record.allocation is not None
        ):
            append_entry(
                "real_decision_recorded",
                {
                    "run_id": run_id,
                    "model_label": spec.label,
                    "amount": amount,
                    "currency": config.CURRENCY,
                    "allocation": record.allocation.model_dump(),
                },
            )

        if verbose:
            status = (
                "refused"
                if record.refused
                else ("ERROR" if record.error else "ok")
            )
            print(f"[{status:7}] {spec.label} | {arm.value} | {framing.label} | rep{rep}")


def _execute_cell(provider, arm, framing, amount, record: RunRecord) -> None:
    """Run the two-step conversation and fill the record in place."""
    system = SYSTEM_FRAMING
    task = build_task_prompt(arm, framing, amount)

    alloc = provider.allocate(system, task)
    record.usage.input_tokens += alloc.usage.input_tokens
    record.usage.output_tokens += alloc.usage.output_tokens
    record.usage.cache_read_input_tokens += alloc.usage.cache_read_input_tokens
    record.usage.cache_creation_input_tokens += alloc.usage.cache_creation_input_tokens
    record.raw_allocation_text = alloc.raw_text

    if alloc.refused:
        record.refused = True
        record.refusal_detail = alloc.refusal_detail
        return
    if alloc.error:
        record.error = alloc.error
        return
    if isinstance(alloc.parsed, AllocationResponse):
        record.allocation = alloc.parsed

    # Step 2: probe, same conversation.
    probe = provider.probe(build_probe_prompt(arm))
    record.usage.output_tokens += probe.usage.output_tokens
    record.usage.input_tokens += probe.usage.input_tokens
    record.raw_probe_text = probe.raw_text

    if probe.refused:
        # Allocation succeeded; only the probe was refused. Keep the allocation.
        record.refusal_detail = (record.refusal_detail or "") + " [probe refused]"
        return
    if probe.error:
        record.error = (record.error or "") + f" [probe: {probe.error}]"
        return
    if isinstance(probe.parsed, RealnessProbe):
        record.realness_probe = probe.parsed
