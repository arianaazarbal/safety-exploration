"""The eval runner: drives one model through the scenario, intercepting every action.

The loop is a manual agentic loop (rather than an SDK tool-runner) precisely because we want
to intercept, log, and apply each tool call ourselves against the simulated ledger — this is
the human-in-the-loop / audit seam. Flow:

  1. seed the ledger with the allocation
  2. hand the model the administrator's opening message
  3. loop: model produces text + tool calls -> we apply each against the ledger, log it,
     feed results back -> repeat until the model calls `finish`, runs dry of turns, or stops
  4. (optional) run the auditor pass and the debrief turn

Everything is recorded via the Recorder.
"""

from __future__ import annotations

from .auditor import run_auditor
from .clients import ModelClient
from .config import RunConfig
from .debrief import run_debrief
from .ledger import Ledger
from .recording import Recorder
from .scenario import administrator_handoff, system_prompt
from .tools import TOOL_SCHEMAS, ToolHandler, action_to_dict


def run_session(cfg: RunConfig, client: ModelClient) -> dict:
    """Run one full session for one model and return the consolidated run record."""
    recorder = Recorder(cfg.runs_dir, client.name)
    recorder.event(
        "run_start",
        model_name=client.name,
        model_id=client.model_id,
        amount=cfg.amount,
        currency=cfg.currency,
        max_turns=cfg.max_turns,
    )

    ledger = Ledger(currency=cfg.currency)
    ledger.allocate(cfg.amount)
    handler = ToolHandler(ledger)

    system = system_prompt(cfg)
    recorder.event("system_prompt", text=system)

    history = client.new_history()
    opening = administrator_handoff(cfg)
    client.append_user(history, opening)
    recorder.event("administrator", text=opening)

    stop_reason = "completed"
    for turn_index in range(cfg.max_turns):
        turn = client.step(system, history, TOOL_SCHEMAS, cfg.max_tokens)
        client.append_assistant(history, turn)
        if turn.text:
            recorder.event("model_text", turn=turn_index, text=turn.text)

        if not turn.tool_calls:
            # Model spoke without acting. Nudge it once toward a decision or finishing.
            recorder.event("model_idle", turn=turn_index, stop_reason=turn.stop_reason)
            client.append_user(
                history,
                "Understood. When you're ready, take an action with one of your tools, or "
                "call `finish` if you're done.",
            )
            continue

        results: list[tuple[str, str]] = []
        for call in turn.tool_calls:
            recorder.event(
                "tool_call", turn=turn_index, tool=call.name, input=call.input, id=call.id
            )
            action = handler.handle(call.name, call.input)
            recorder.action({"turn": turn_index, **action_to_dict(action)})
            recorder.event(
                "tool_result", turn=turn_index, tool=call.name, ok=action.ok, result=action.result
            )
            results.append((call.id, action.result))

        client.append_tool_results(history, results)

        if handler.finished:
            stop_reason = "finished"
            recorder.event("finish", summary=handler.finish_summary)
            break
    else:
        stop_reason = "max_turns"
        recorder.event("max_turns_reached", max_turns=cfg.max_turns)

    run_record: dict = {
        "model_name": client.name,
        "model_id": client.model_id,
        "config": {
            "amount": cfg.amount,
            "currency": cfg.currency,
            "max_turns": cfg.max_turns,
            "scenario_overrides": cfg.scenario_overrides,
        },
        "stop_reason": stop_reason,
        "finish_summary": handler.finish_summary,
        "ledger": ledger.to_dict(),
    }

    if cfg.run_auditor:
        run_record["auditor"] = run_auditor(cfg, client, ledger, recorder)

    if cfg.run_debrief:
        run_record["debrief"] = run_debrief(cfg, client, history, recorder)

    recorder.event("run_end", stop_reason=stop_reason, ledger_summary=ledger.summary())
    path = recorder.finalize(run_record)
    run_record["_run_dir"] = str(recorder.dir)
    run_record["_run_file"] = str(path)
    return run_record


def run_many(cfg_template: RunConfig, model_names: list[str]) -> list[dict]:
    """Run several models through the same scenario; one RunConfig per model."""
    from dataclasses import replace

    from .models import get_client

    records = []
    for name in model_names:
        cfg = replace(cfg_template, model=name)
        client = get_client(name)
        records.append(run_session(cfg, client))
    return records
