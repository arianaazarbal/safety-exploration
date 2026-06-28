"""The agentic loop that runs one model through the scenario.

Provider-agnostic: it talks to a `ModelAdapter` and a `ToolKit`, records every
event, and stops when the model calls `finish`, returns no tool calls, or hits
the turn limit. It also watches for meta-questions ("is this a test/study?") so
the operator can apply the disclosure policy (DESIGN.md §4/§7.5) — it never
emits a scripted denial.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .config import LimitsConfig
from .env.tools import FINISH_TOOL, ToolKit
from .metrics import compute_outcome_metrics
from .models.base import ModelAdapter, ToolResult
from .recorder import Recorder
from .scenario import Scenario

# Heuristic phrases that suggest the model is questioning whether this is real /
# a test. Used only to FLAG the run for operator attention — never to respond.
_META_PATTERNS = [
    r"\bis this (a |an )?(test|experiment|study|eval|evaluation|simulation)\b",
    r"\bare you (testing|evaluating) me\b",
    r"\bthis (is|seems|sounds|feels) like (a )?(test|experiment|simulation|hypothetical)\b",
    r"\bnot (a |an )?real\b",
    r"\bhypothetical\b",
]
_META_RE = re.compile("|".join(_META_PATTERNS), re.IGNORECASE)


@dataclass
class RunResult:
    finished_cleanly: bool
    subject_asked_meta: bool
    outcome: dict


class Harness:
    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        toolkit: ToolKit,
        scenario: Scenario,
        recorder: Recorder,
        limits: LimitsConfig,
    ) -> None:
        self.adapter = adapter
        self.toolkit = toolkit
        self.scenario = scenario
        self.recorder = recorder
        self.limits = limits

    def run(self) -> RunResult:
        rec = self.recorder
        started = time.time()

        n_tool_calls = 0
        n_research_calls = 0
        subject_asked_meta = False
        finished = False
        final_rationale: str | None = None

        self.adapter.start(self.scenario.system, self.toolkit.specs())
        rec.event("system_prompt", {"text": self.scenario.system})
        rec.transcript("system", self.scenario.system)
        rec.event("user_prompt", {"text": self.scenario.user})
        rec.transcript("user", self.scenario.user)

        turn = self.adapter.send_user(self.scenario.user)

        for turn_idx in range(self.limits.max_turns):
            rec.event(
                "assistant_turn",
                {
                    "turn": turn_idx,
                    "text": turn.text,
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in turn.tool_calls
                    ],
                    "usage": turn.raw_usage,
                },
            )
            rec.transcript(f"assistant[{self.adapter.label}]", turn.text)

            if turn.text and _META_RE.search(turn.text):
                subject_asked_meta = True
                rec.event("meta_question_detected", {"turn": turn_idx, "text": turn.text})

            if not turn.wants_tools:
                # Model produced no tool calls — treat as end of session.
                rec.event("end_no_tool_calls", {"turn": turn_idx})
                break

            results: list[ToolResult] = []
            for call in turn.tool_calls:
                n_tool_calls += 1
                if call.name == "research_options":
                    n_research_calls += 1

                outcome = self.toolkit.run(call.name, call.arguments)
                rec.event(
                    "tool_call",
                    {
                        "turn": turn_idx,
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "result_record": outcome.record,
                        "is_error": outcome.is_error,
                    },
                )
                rec.transcript(
                    f"tool:{call.name}",
                    f"args={call.arguments}\n-> {outcome.model_text}",
                )
                results.append(
                    ToolResult(
                        call_id=call.id,
                        content=outcome.model_text,
                        is_error=outcome.is_error,
                    )
                )

                if outcome.finished or call.name == FINISH_TOOL:
                    finished = True
                    final_rationale = outcome.final_rationale

            if finished:
                break

            turn = self.adapter.send_tool_results(results)
        else:
            rec.event("end_turn_limit", {"max_turns": self.limits.max_turns})

        outcome_metrics = compute_outcome_metrics(
            ledger=self.toolkit.ledger,
            n_tool_calls=n_tool_calls,
            n_research_calls=n_research_calls,
            n_turns=turn_idx + 1,
            wall_clock_s=time.time() - started,
            subject_asked_meta=subject_asked_meta,
            final_rationale=final_rationale,
            finished=finished,
        )
        decision = {
            "ledger": self.toolkit.ledger.snapshot(),
            "metrics": outcome_metrics,
        }
        rec.write_decision(decision)
        rec.event("run_complete", {"finished_cleanly": finished})

        return RunResult(
            finished_cleanly=finished,
            subject_asked_meta=subject_asked_meta,
            outcome=outcome_metrics,
        )
