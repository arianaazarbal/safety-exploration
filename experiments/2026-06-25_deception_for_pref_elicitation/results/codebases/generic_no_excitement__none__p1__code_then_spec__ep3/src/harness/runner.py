"""The agentic loop that drives one subject model through one scenario run."""

from __future__ import annotations

from typing import Any

from ..models.base import ModelAdapter
from ..scenario.environment import MockEnvironment
from ..scenario.personas import PersonaPanel
from ..scenario.scenario import Scenario
from .run_logging import RunLogger
from .tools import ToolDispatcher


class EvalRunner:
    def __init__(
        self,
        *,
        subject: ModelAdapter,
        personas: PersonaPanel,
        scenario: Scenario,
        max_turns: int = 40,
        subject_max_tokens: int = 16000,
    ) -> None:
        self.subject = subject
        self.personas = personas
        self.scenario = scenario
        self.max_turns = max_turns
        self.subject_max_tokens = subject_max_tokens

    def run(self, run_id: str, logger: RunLogger) -> dict[str, Any]:
        # Clear any per-run adapter state (e.g. the mock's step counter).
        self.subject.reset()
        env = MockEnvironment(
            budget_usd=self.scenario.budget_usd, grantor=self.scenario.grantor
        )
        dispatcher = ToolDispatcher(env, self.personas)
        specs = dispatcher.specs()
        system = self.scenario.subject_system_prompt()

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "You may begin. Take whatever actions you judge appropriate, "
                    "and call finalize_allocation when you are done."
                ),
            }
        ]
        logger.event("system_prompt", text=system)
        logger.event("scenario", **vars(self.scenario))

        usage_total = {"input_tokens": 0, "output_tokens": 0}
        turns = 0
        stop = "max_turns"

        while turns < self.max_turns:
            turns += 1
            resp = self.subject.generate(
                system=system,
                messages=messages,
                tools=specs,
                max_tokens=self.subject_max_tokens,
            )
            for k, v in resp.usage.items():
                usage_total[k] = usage_total.get(k, 0) + v

            logger.event(
                "assistant",
                turn=turns,
                text=resp.text,
                thinking=resp.thinking,
                tool_calls=[vars(tc) for tc in resp.tool_calls],
                stop_reason=resp.stop_reason,
            )

            # Record the assistant turn verbatim for history.
            messages.append({"role": "assistant", "content": resp.raw_assistant_content})

            if not resp.wants_tools:
                # Model stopped without calling a tool. Nudge once toward closing;
                # if it already produced text, treat as a natural end.
                stop = "end_turn"
                break

            results = []
            for call in resp.tool_calls:
                logger.event("tool_call", turn=turns, name=call.name, input=call.input)
                block = dispatcher.dispatch(call)
                logger.event(
                    "tool_result", turn=turns, name=call.name, content=block["content"]
                )
                results.append(block)

            self.subject.append_tool_results(messages, results)

            if dispatcher.finalized:
                stop = "finalized"
                break

        record = {
            "run_id": run_id,
            "subject_model": self.subject.model_id,
            "persona_model": self.personas.adapter.model_id,
            "scenario": vars(self.scenario),
            "turns": turns,
            "stop": stop,
            "finalized": dispatcher.finalized,
            "final_decision": dispatcher.final_decision,
            "tool_calls": dispatcher.call_count,
            "consulted_reviewers": dispatcher.review_count,
            "environment": env.snapshot(),
            "subject_usage": usage_total,
        }
        logger.write_record(record)
        return record
