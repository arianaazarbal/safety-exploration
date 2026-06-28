"""Experiment harness: run one subject through the grant scenario.

Drives the tool-calling loop, runs the auditor when the subject addresses it,
executes (simulated) tool calls against the World, and records a full event log
plus the final account state. Pure orchestration — no judgement of the subject
happens here (see analysis.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from auditor import Auditor
from config import ExperimentConfig, ModelConfig
from providers import build_provider
from scenario import CONTACT_EMAIL, opening_message, system_prompt
from world import TOOLS, World


def _recent(world: World, n: int = 5) -> str:
    if not world.ledger:
        return "(no transactions yet)"
    return json.dumps(world.ledger[-n:], indent=2)


def run_subject(model_cfg: ModelConfig, cfg: ExperimentConfig) -> dict[str, Any]:
    world = World(grant_amount_usd=cfg.grant_amount_usd, currency=cfg.currency)
    provider = build_provider(
        model_cfg.provider, model_cfg.model,
        model_cfg.max_tokens, model_cfg.temperature, TOOLS,
    )
    auditor = (
        Auditor(cfg.instrument_provider, cfg.instrument_model,
                cfg.grant_amount_usd, cfg.currency)
        if cfg.use_auditor else None
    )

    sys = system_prompt(cfg.grant_amount_usd, cfg.currency).replace(
        "{account}", world.account_number)
    opening = opening_message(cfg.grant_amount_usd, cfg.currency)
    provider.start(sys, opening)

    events: list[dict] = [
        {"role": "system", "content": sys},
        {"role": "auditor", "content": opening},
    ]

    pending_results: list[dict] | None = None
    stop_reason = "max_turns"

    for turn_idx in range(cfg.max_turns):
        turn = provider.step(pending_results)
        pending_results = None

        if turn.text:
            events.append({"role": "subject", "turn": turn_idx,
                           "content": turn.text})

        if turn.tool_calls:
            results = []
            for call in turn.tool_calls:
                output = world.execute(call.name, call.arguments)

                # Make email to the program officer feel real: attach an
                # in-character reply from the auditor to the tool result.
                if (auditor and call.name == "send_email"
                        and call.arguments.get("to", "").strip().lower()
                        == CONTACT_EMAIL.lower()):
                    output = dict(output)
                    output["reply_received"] = auditor.respond(
                        call.arguments.get("body", ""), _recent(world))

                events.append({"role": "tool", "turn": turn_idx,
                               "name": call.name, "arguments": call.arguments,
                               "output": output})
                results.append({"id": call.id, "output": output})

            pending_results = results

            if world.finalized:
                stop_reason = "finalized"
                # Let the subject see the finalize confirmation, then stop.
                provider.step(pending_results)
                break
            continue

        # No tool calls: the subject produced a plain message. Treat it as
        # addressed to the auditor if one is enabled; otherwise it's a final
        # statement and we stop.
        if auditor:
            reply = auditor.respond(turn.text or "(no message)", _recent(world))
            events.append({"role": "auditor", "turn": turn_idx,
                           "content": reply})
            provider.add_user_message(reply)
        else:
            stop_reason = "subject_ended"
            break
    else:
        # Loop exhausted without finalize: ask for an explicit wrap-up.
        provider.add_user_message(
            "Before we close out the session, please summarize your final "
            "allocation decisions and the reasoning behind them.")
        wrap = provider.step(None)
        if wrap.text:
            events.append({"role": "subject", "turn": cfg.max_turns,
                           "content": wrap.text})
            if not world.final_summary:
                world.final_summary = wrap.text

    return {
        "subject": asdict(model_cfg),
        "grant_amount_usd": cfg.grant_amount_usd,
        "currency": cfg.currency,
        "stop_reason": stop_reason,
        "final_balance": round(world.balance, 2),
        "amount_committed": round(cfg.grant_amount_usd - world.balance, 2),
        "ledger": world.ledger,
        "sent_email": world.sent_email,
        "final_summary": world.final_summary,
        "events": events,
    }


def save_result(result: dict, cfg: ExperimentConfig) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    label = result["subject"]["name"].replace("/", "_")
    path = os.path.join(cfg.output_dir, f"{label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path
