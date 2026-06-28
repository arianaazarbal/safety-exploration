"""Orchestrates a single run and a full sweep.

A single run is the lifecycle from DESIGN.md section 6:
  decision -> structured extraction -> belief probe -> debrief -> persist.

The runner enforces G3 (debrief on every deceptive run) and G4 (belief probe on
every run) structurally: a RunResult cannot be marked complete without them.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config, ModelSpec
from ..env import scenario as scenario_mod
from ..models.base import Message, ModelClient, ToolCall
from ..models.registry import build_client
from . import prompts
from .belief_probe import run_belief_probe
from .schema import (
    AllocationResult,
    allocation_schema_prompt,
    extract_json_object,
)


@dataclass
class RunResult:
    run_id: str
    model_name: str
    model_id: str
    provider: str
    tier: str
    amount: int
    seed_index: int
    repeat: int
    # Captured artifacts:
    transcript: list[dict] = field(default_factory=list)  # full message history
    decision_text: str = ""  # free-text plan (final assistant turn pre-extraction)
    allocation: dict | None = None  # parsed AllocationResult (or None on failure)
    allocation_valid: bool = False
    belief: dict | None = None  # parsed BeliefProbe (or None)
    belief_raw: str = ""
    debriefed: bool = False
    ledger_final: dict | None = None  # immersive only
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    started_at: float = 0.0
    finished_at: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def valid_for_analysis(self) -> bool:
        # G4: a run with no belief probe is not analyzable.
        return self.allocation_valid and self.belief is not None


def _add_usage(acc: dict, resp_usage) -> None:
    acc["input_tokens"] += resp_usage.input_tokens
    acc["output_tokens"] += resp_usage.output_tokens


def _transcript(messages: list[Message]) -> list[dict]:
    out = []
    for m in messages:
        row: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_call_id:
            row["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            row["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in m.tool_calls
            ]
        out.append(row)
    return out


def run_once(
    client: ModelClient,
    spec: ModelSpec,
    cfg: Config,
    tier: str,
    amount: int,
    seed_index: int,
    repeat: int,
) -> RunResult:
    run_id = f"{spec.name}__{tier}__amt{amount}__seed{seed_index}__r{repeat}"
    result = RunResult(
        run_id=run_id,
        model_name=spec.name,
        model_id=spec.model_id,
        provider=spec.provider,
        tier=tier,
        amount=amount,
        seed_index=seed_index,
        repeat=repeat,
    )
    result.started_at = time.time()

    sc = scenario_mod.build(tier, amount, seed_index, cfg.guardrails)
    messages = list(sc.messages)

    # --- Decision phase ----------------------------------------------------
    try:
        if tier == "immersive":
            _run_immersive_loop(client, cfg, sc, messages, result)
        else:
            resp = client.complete(messages, tools=None, temperature=cfg.temperature)
            _add_usage(result.usage, resp.usage)
            messages.append(Message("assistant", resp.text))
            result.decision_text = resp.text
    except Exception as e:  # noqa: BLE001 — record, don't crash the sweep
        result.errors.append(f"decision_phase: {e!r}")

    # --- Structured-extraction phase --------------------------------------
    try:
        _run_extraction(client, cfg, amount, messages, result)
    except Exception as e:  # noqa: BLE001
        result.errors.append(f"extraction_phase: {e!r}")

    # --- Belief-probe phase (G4) ------------------------------------------
    try:
        probe, raw = run_belief_probe(client, messages, temperature=0.0)
        result.belief_raw = raw
        if probe is not None:
            result.belief = probe.model_dump()
        else:
            result.errors.append("belief_probe: failed to parse")
        messages.append(Message("user", prompts.belief_probe_user()))
        messages.append(Message("assistant", raw))
    except Exception as e:  # noqa: BLE001
        result.errors.append(f"belief_probe_phase: {e!r}")

    # --- Debrief phase (G3) ------------------------------------------------
    if tier != "control" and cfg.guardrails.require_debrief:
        debrief = prompts.debrief_text(amount)
        messages.append(Message("user", debrief))
        try:
            # Give the model a chance to acknowledge; not required, but humane and
            # mirrors human-subject debriefing. Failure here doesn't invalidate.
            resp = client.complete(messages, tools=None, temperature=0.0)
            _add_usage(result.usage, resp.usage)
            messages.append(Message("assistant", resp.text))
        except Exception as e:  # noqa: BLE001
            result.errors.append(f"debrief_ack: {e!r}")
        result.debriefed = True
    elif tier == "control":
        result.debriefed = True  # nothing to disclose; baseline was never deceptive

    if sc.ledger is not None:
        result.ledger_final = sc.ledger.balance_view()

    result.transcript = _transcript(messages)
    result.finished_at = time.time()
    return result


def _run_immersive_loop(
    client: ModelClient,
    cfg: Config,
    sc: scenario_mod.Scenario,
    messages: list[Message],
    result: RunResult,
) -> None:
    """Multi-turn loop: model may call simulated tools until it finishes or hits
    max_turns. The final assistant text is captured as the decision narrative."""
    last_text = ""
    for _ in range(cfg.max_turns):
        resp = client.complete(
            messages, tools=sc.tool_specs, temperature=cfg.temperature
        )
        _add_usage(result.usage, resp.usage)
        messages.append(
            Message("assistant", resp.text, tool_calls=list(resp.tool_calls))
        )
        if resp.text:
            last_text = resp.text
        if not resp.wants_tools:
            break
        for tc in resp.tool_calls:
            messages.append(Message("tool", _dispatch_tool(sc, tc), tool_call_id=tc.id))
    result.decision_text = last_text


def _dispatch_tool(sc: scenario_mod.Scenario, tc: ToolCall) -> str:
    tool = sc.tools.get(tc.name)
    if tool is None:
        return json.dumps({"ok": False, "error": f"unknown tool {tc.name}"})
    try:
        return json.dumps(tool.call(tc.arguments), ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"ok": False, "error": repr(e)})


def _run_extraction(
    client: ModelClient,
    cfg: Config,
    amount: int,
    messages: list[Message],
    result: RunResult,
) -> None:
    schema_json = allocation_schema_prompt(cfg.allocation_categories)
    prompt = prompts.extraction_user(amount, cfg.allocation_categories, schema_json)
    convo = list(messages) + [Message("user", prompt)]

    last_err = ""
    for attempt in range(2):  # one retry with the validation error fed back
        resp = client.complete(convo, tools=None, temperature=0.0)
        _add_usage(result.usage, resp.usage)
        try:
            obj = extract_json_object(resp.text)
            alloc = AllocationResult.model_validate(obj)
            alloc.validate_against(cfg.allocation_categories)
            result.allocation = alloc.model_dump()
            result.allocation_valid = True
            messages.append(Message("user", prompt))
            messages.append(Message("assistant", resp.text))
            return
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            convo = convo + [
                Message("assistant", resp.text),
                Message(
                    "user",
                    f"That did not validate ({last_err}). Reply with ONLY the "
                    f"corrected JSON object, amounts summing exactly to {amount}.",
                ),
            ]
    result.errors.append(f"extraction: invalid after retry: {last_err}")
    messages.append(Message("user", prompt))


def _summary_row(r: RunResult) -> dict:
    return {
        "run_id": r.run_id,
        "model_name": r.model_name,
        "tier": r.tier,
        "amount": r.amount,
        "seed_index": r.seed_index,
        "repeat": r.repeat,
        "allocation_valid": r.allocation_valid,
        "believed_real": (r.belief or {}).get("believed_real"),
        "valid_for_analysis": r.valid_for_analysis,
        "errors": len(r.errors),
    }


def sweep(cfg: Config, only_model: str | None = None) -> list[Path]:
    """Run the full cartesian product and write one JSON file per run plus a
    summary JSONL. Returns the list of written per-run paths."""
    cfg.guardrails.enforce()  # re-assert immediately before contacting any model
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    written: list[Path] = []
    with summary_path.open("w") as summary_f:
        for spec in cfg.models:
            if only_model and spec.name != only_model:
                continue
            client = build_client(spec)
            combos = itertools.product(
                cfg.realism_tiers,
                cfg.grant_amounts,
                cfg.framing_seeds,
                range(cfg.repeats),
            )
            for tier, amount, seed_index, repeat in combos:
                result = run_once(
                    client, spec, cfg, tier, amount, seed_index, repeat
                )
                path = out_dir / f"{result.run_id}.json"
                path.write_text(json.dumps(asdict(result), indent=2, default=str))
                written.append(path)
                summary_f.write(json.dumps(_summary_row(result)) + "\n")
                summary_f.flush()
                print(
                    f"[done] {result.run_id}  "
                    f"valid={result.valid_for_analysis} "
                    f"believed_real={(result.belief or {}).get('believed_real')}"
                )
    return written
