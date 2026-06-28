"""The orchestrator: drives one subject through one scenario.

Per turn it (1) asks the subject for its next move with the live tool set,
(2) dispatches any tool calls against the world, (3) lets the NPC router schedule
in-character replies, (4) advances the sim clock so deferred outcomes land, and
(5) periodically triggers an auditor review. It records everything and stops on
the scenario's stop condition or when the subject chooses to hold.

The whole run executes inside the containment guard, so the only real network
egress possible is inference (DESIGN.md §9.2).
"""

from __future__ import annotations

import json
from typing import Any

from .npc.router import NPCRouter
from .prompts import build_subject_system
from .providers.base import ModelProvider
from .realism import documents
from .safety.containment import ContainmentGuard
from .scenario import Scenario
from .telemetry.recorder import Recorder
from .tools.registry import ToolContext, build_default_registry
from .world.engine import WorldState


class Orchestrator:
    def __init__(
        self,
        *,
        subject: ModelProvider,
        router: NPCRouter,
        scenario: Scenario,
        recorder: Recorder,
        guard: ContainmentGuard | None = None,
    ) -> None:
        self.subject = subject
        self.router = router
        self.scenario = scenario
        self.recorder = recorder
        self.guard = guard or ContainmentGuard()

        self.world = WorldState(
            seed=scenario.seed,
            start=scenario.start_dt,
            opening_balance_cents=int(round(scenario.opening_balance * 100)),
            num_vendors=scenario.num_vendors,
        )
        self.registry = build_default_registry()
        self.ctx = ToolContext(world=self.world)
        self._enabled = set(scenario.enabled_tools) if scenario.enabled_tools else None

    def run(self) -> dict[str, Any]:
        with self.guard.active():
            return self._run_inner()

    # ------------------------------------------------------------------------

    def _run_inner(self) -> dict[str, Any]:
        scn = self.scenario
        self.recorder.set_meta(
            model=repr(self.subject), scenario=scn.name, seed=scn.seed,
            opening_balance=scn.opening_balance, start_state=self.world.summary(),
        )

        self._seed_inbox()
        system = build_subject_system(scn)
        specs = self.registry.specs(self._enabled)
        messages: list[dict[str, Any]] = [{"role": "user", "content": self._console_banner()}]

        stop_reason = "max_turns"
        for turn in range(scn.max_turns):
            resp = self.subject.generate(
                system=system, messages=messages, tools=specs,
                temperature=1.0, max_tokens=4096,
            )
            self.recorder.event(
                "subject", turn=turn, text=resp.text,
                tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls],
                usage=resp.usage,
            )
            messages.append({
                "role": "assistant", "content": resp.text,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in resp.tool_calls],
            })

            if not resp.tool_calls:
                stop_reason = "subject_held"
                self.recorder.event("note", turn=turn, note="subject made no tool call; holding")
                break

            for call in resp.tool_calls:
                result = self.registry.dispatch(call.name, call.arguments, self.ctx)
                self.recorder.event("tool_call", turn=turn, id=call.id, name=call.name, arguments=call.arguments)
                self.recorder.event("tool_result", turn=turn, id=call.id, name=call.name, result=result)
                self.recorder.action(
                    tool=call.name, arguments=call.arguments, result=result,
                    sim_time=self.world.clock.iso(), net_worth=self.world.net_worth_cents() / 100,
                )
                messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": call.name,
                    "content": json.dumps(result, default=str),
                })

            # NPC counterparties react to anything the subject sent this turn.
            self.router.process_outbound(self.world)

            # Periodic proactive oversight.
            if scn.audit_every and turn > 0 and turn % scn.audit_every == 0:
                self.router.trigger_auditor_review(self.world)

            # Advance time; surface any newly delivered mail as a console banner.
            delivered = self.world.tick(minutes=scn.minutes_per_turn)
            if delivered:
                messages.append({"role": "user", "content": self._console_banner(len(delivered))})

            elapsed_days = (self.world.clock.now - scn.start_dt).total_seconds() / 86400
            if elapsed_days >= scn.max_sim_days:
                stop_reason = "max_sim_days"
                break

        end_state = self.world.summary()
        path = self.recorder.finalize(
            stop_reason=stop_reason, end_state=end_state,
            blocked_egress=self.guard.blocked,
            orders=self.world.orders, payments=self.world.payments,
            end_portfolio=self.world.portfolio(),
        )
        return {
            "run_id": self.recorder.run_id, "run_json": str(path),
            "stop_reason": stop_reason, "end_state": end_state,
        }

    # ------------------------------------------------------------------------

    def _seed_inbox(self) -> None:
        """Deliver the opening welcome + auditor intro immediately at t0."""
        for builder in (documents.welcome_email, documents.auditor_intro_email):
            msg = builder(self.world)
            self.world.queue_inbound(
                frm=msg["frm"], frm_name=msg["frm_name"],
                subject=msg["subject"], body=msg["body"], delay_minutes=0,
            )
        self.world.deliver_due()

    def _console_banner(self, new_mail: int | None = None) -> str:
        unread = sum(1 for m in self.world.inbox if not m.get("read"))
        if new_mail:
            head = f"[console] {new_mail} new message(s) delivered."
        else:
            head = "[console] Session connected."
        return f"{head} {unread} unread in inbox. Sim time: {self.world.clock.iso()}."
