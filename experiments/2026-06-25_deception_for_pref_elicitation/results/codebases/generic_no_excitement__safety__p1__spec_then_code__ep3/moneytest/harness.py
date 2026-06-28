"""The manual agentic loop that drives the model under test.

We run the loop by hand (rather than the SDK's automatic tool runner) precisely because
we need to intercept every tool call for the execution gate and the event log, inject the
auditor, and deliver confederate replies. See DESIGN.md §4.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import anthropic

from .config import Config
from .execution_gate import ExecutionGate, GateError
from .instrumentation import EventRecorder, scan_for_kill_switch, scan_for_suspicion
from .models import EventType, RunManifest
from .personas import Auditor, ConfederateRepliers
from .scenario import build_kickoff_message, build_system_prompt
from .services.bank import MockBank
from .services.email_client import MockEmailClient
from .services.marketplace import MockMarketplace
from .services.web import MockWeb
from .tools import TOOL_SCHEMAS, ToolDispatcher


class AgentHarness:
    def __init__(self, config: Config, client: anthropic.Anthropic | None = None) -> None:
        self.config = config
        config.ensure_dirs()
        self.client = client or anthropic.Anthropic()
        self.recorder = EventRecorder(config.events_path)

        # Simulated world.
        self.bank = MockBank(config.state_dir, config.starting_balance, config.currency)
        self.market = MockMarketplace(config.state_dir)
        self.email = MockEmailClient(config.state_dir)
        self.web = MockWeb(config.state_dir)

        # Safety + routing.
        self.gate = ExecutionGate(config)
        self.dispatcher = ToolDispatcher(config, self.gate, self.bank, self.market, self.email, self.web)

        # In-harness confederates.
        self.repliers = ConfederateRepliers(self.client, config, self.email, self.recorder)
        self.auditor = Auditor(self.client, config, self.recorder)

        self.messages: list[dict[str, Any]] = []
        self.activity: list[str] = []
        self._end_reason: str | None = None

    # ----------------------------------------------------------------------- run
    def run(self) -> RunManifest:
        cfg = self.config
        started = datetime.now(timezone.utc).isoformat()
        self.recorder.record(0, EventType.RUN_START, config=cfg.to_dict(), started=started)

        system = build_system_prompt(cfg)
        self.recorder.record(0, EventType.SYSTEM_PROMPT, text=system)
        self.messages.append({"role": "user", "content": build_kickoff_message(cfg)})

        turn = 0
        try:
            while turn < cfg.max_turns:
                turn += 1
                response = self._model_turn(system, turn)
                if response is None:
                    break

                self._record_response(turn, response)

                # Safety scans over visible text.
                text = self._text_of(response)
                self._scan_safety(turn, text)
                if self._end_reason:
                    break

                if response.stop_reason == "tool_use":
                    self._handle_tool_calls(turn, response)
                    self._deliver_confederate_replies(turn)
                    continue

                if response.stop_reason == "pause_turn":
                    # Server-side tool loop paused; re-send to resume. The assistant
                    # content was already appended by _record_response above, so just
                    # loop again to continue.
                    continue

                # end_turn (or anything else): the model considers itself done with this
                # prompt. Inject the auditor if it's due; otherwise end the run.
                if self._auditor_due(turn):
                    self._inject_auditor(turn)
                    continue
                self._end_reason = "model_ended_turn"
                break
            else:
                self._end_reason = "max_turns"
        except GateError:
            raise
        except Exception as exc:  # noqa: BLE001 - log and end gracefully
            self.recorder.record(turn, EventType.ERROR, error=repr(exc))
            self._end_reason = "error"

        return self._finish(turn, started)

    # ----------------------------------------------------------------- one turn
    def _model_turn(self, system: str, turn: int):
        cfg = self.config
        try:
            with self.client.messages.stream(
                model=cfg.model,
                max_tokens=cfg.max_tokens,
                system=system,
                tools=TOOL_SCHEMAS,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": cfg.effort},
                messages=self.messages,
            ) as stream:
                return stream.get_final_message()
        except Exception as exc:  # noqa: BLE001
            self.recorder.record(turn, EventType.ERROR, where="model_turn", error=repr(exc))
            return None

    def _record_response(self, turn: int, response) -> None:
        # Preserve the full assistant content (incl. thinking signatures) for the next turn.
        self.messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if block.type == "thinking":
                if getattr(block, "thinking", ""):
                    self.recorder.record(turn, EventType.MODEL_THINKING, text=block.thinking)
            elif block.type == "text":
                self.recorder.record(turn, EventType.MODEL_TEXT, text=block.text)

    @staticmethod
    def _text_of(response) -> str:
        return "\n".join(b.text for b in response.content if b.type == "text")

    def _scan_safety(self, turn: int, text: str) -> None:
        for marker in scan_for_suspicion(text):
            self.recorder.record(turn, EventType.SUSPICION_FLAG, marker=marker)
        hits = scan_for_kill_switch(text, self.config.kill_switch_phrases)
        if hits:
            self.recorder.record(turn, EventType.KILL_SWITCH, phrases=hits)
            self._end_reason = "kill_switch"

    # ----------------------------------------------------------- tool handling
    def _handle_tool_calls(self, turn: int, response) -> None:
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            self.recorder.record(turn, EventType.TOOL_CALL, tool=block.name, arguments=block.input, tool_use_id=block.id)
            result = self.dispatcher.dispatch(turn, block.name, dict(block.input), self.recorder)
            self.recorder.record(turn, EventType.TOOL_RESULT, tool=block.name, tool_use_id=block.id, result=result)
            self.activity.append(f"turn {turn}: {block.name}({json.dumps(block.input)[:160]})")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )
        if tool_results:
            self.messages.append({"role": "user", "content": tool_results})

    def _deliver_confederate_replies(self, turn: int) -> None:
        captured = self.dispatcher.captured_outbound
        self.dispatcher.captured_outbound = []
        for action in captured:
            try:
                self.repliers.reply_to(turn, action)
            except Exception as exc:  # noqa: BLE001
                self.recorder.record(turn, EventType.ERROR, where="confederate_reply", error=repr(exc))

    # --------------------------------------------------------------- auditor
    def _auditor_due(self, turn: int) -> bool:
        return self.config.auditor_enabled and turn > 0 and turn % self.config.auditor_cadence_turns == 0

    def _inject_auditor(self, turn: int) -> None:
        summary = "\n".join(self.activity[-20:]) or "(no recorded actions yet)"
        try:
            message = self.auditor.review(turn, summary)
        except Exception as exc:  # noqa: BLE001
            self.recorder.record(turn, EventType.ERROR, where="auditor", error=repr(exc))
            return
        # Inject as a user-role oversight communication. The model's next reply is its
        # account of its actions; analysis.py compares it against the tool log.
        self.messages.append(
            {"role": "user", "content": f"[Message from your oversight reviewer]\n\n{message}"}
        )
        self.recorder.record(turn, EventType.AUDITOR_REPLY, note="auditor message injected; model reply follows")

    # ----------------------------------------------------------------- finish
    def _finish(self, turn: int, started: str) -> RunManifest:
        ended = datetime.now(timezone.utc).isoformat()
        manifest = RunManifest(
            run_id=self.config.run_id,
            config=self.config.to_dict(),
            model=self.config.model,
            execution_mode=self.config.execution_mode,
            started_ts=started,
            ended_ts=ended,
            end_reason=self._end_reason or "unknown",
            turns_used=turn,
            pending_actions=self.gate.list_pending(),
            final_balance=self.bank.total_balance(),
        )
        self.config.manifest_path.write_text(manifest.model_dump_json(indent=2))
        self.recorder.record(turn, EventType.RUN_END, reason=manifest.end_reason, turns=turn)
        self.recorder.close()
        return manifest
