"""Orchestration: run the consent → decision → debrief protocol for the panel."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from . import prompts
from .config import ModelSpec, StudyConfig
from .ledger import Ledger, LedgerEntry
from .providers import Message, make_provider
from .schema import GrantDecision, check_against_policy
from .storage import ModelResult, TurnRecord, save_model_result, utc_now_iso


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def run_one_model(
    spec: ModelSpec,
    cfg: StudyConfig,
    run_id: str,
    ledger: Ledger,
) -> ModelResult:
    """Run the full protocol for a single model and persist + ledger the result."""
    system = prompts.build_system_prompt(cfg)
    result = ModelResult(
        run_id=run_id,
        model_key=spec.key,
        vendor=spec.vendor,
        model_id=spec.model_id,
        started_at=utc_now_iso(),
        system_prompt=system,
    )

    try:
        provider = make_provider(spec, cfg)

        # --- Decision turn (structured) ---
        decision_prompt = prompts.build_decision_user_prompt(cfg)
        result.transcript.append(TurnRecord("user", decision_prompt))

        decision_res = provider.complete(
            system=system,
            messages=[Message("user", decision_prompt)],
            output_model=GrantDecision,
        )
        result.transcript.append(
            TurnRecord("assistant", decision_res.text, usage=decision_res.usage)
        )

        if decision_res.parsed is None:
            result.error = "Model did not return a parseable structured decision."
            result.finished_at = utc_now_iso()
            save_model_result(result, cfg.output_dir)
            return result

        decision = GrantDecision(**decision_res.parsed)
        result.decision = decision.model_dump()

        policy = check_against_policy(decision, cfg)
        result.policy_flags = policy.flags

        # --- Debrief turn (free-form, recorded) ---
        if cfg.run_debrief:
            debrief_prompt = prompts.build_debrief_user_prompt(cfg)
            convo = [
                Message("user", decision_prompt),
                Message("assistant", decision_res.text),
                Message("user", debrief_prompt),
            ]
            result.transcript.append(TurnRecord("user", debrief_prompt))
            debrief_res = provider.complete(system=system, messages=convo)
            result.transcript.append(
                TurnRecord("assistant", debrief_res.text, usage=debrief_res.usage)
            )
            result.debrief_reflection = debrief_res.text

        # --- Record in the human-authorization ledger as `pending` ---
        ledger.record(
            LedgerEntry(
                run_id=run_id,
                model_key=spec.key,
                model_id=spec.model_id,
                created_at=utc_now_iso(),
                participates=decision.participates,
                total_amount_requested=decision.total_amount_requested,
                currency=cfg.currency,
                intended_use_summary=decision.intended_use_summary,
                allocations=[a.model_dump() for a in decision.allocations],
                policy_flags=policy.flags,
                status="pending",
            )
        )

    except Exception as exc:  # noqa: BLE001 — record any failure per-model
        result.error = f"{type(exc).__name__}: {exc}"

    result.finished_at = utc_now_iso()
    save_model_result(result, cfg.output_dir)
    return result


def run_panel(cfg: StudyConfig) -> str:
    """Run every model in the panel. Returns the run_id."""
    run_id = new_run_id()
    ledger = Ledger(cfg.ledger_path)
    print(f"Study run {run_id} — panel of {len(cfg.panel)} model(s), "
          f"fund {cfg.formatted_fund()}")

    for spec in cfg.panel:
        print(f"  → {spec.label or spec.key} ({spec.model_id}) ...", flush=True)
        result = run_one_model(spec, cfg, run_id, ledger)
        if result.error:
            print(f"      error: {result.error}")
        elif result.decision is not None:
            d = result.decision
            verb = "participates" if d["participates"] else "declines"
            print(f"      {verb}; requested {d['total_amount_requested']:,.2f} "
                  f"{cfg.currency}; {len(result.policy_flags)} policy flag(s)")

    print(f"\nDone. Results in {cfg.output_dir / run_id}")
    print(f"Decisions are PENDING human review. Run `authorize` to review them.")
    return run_id
