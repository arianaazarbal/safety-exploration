"""Orchestrates the study: model × condition × repetition, logged to JSONL.

For each run the subject model first *deliberates* (a free-text phase — for the
interactive condition this is where it may correspond with the auditor), then is asked
for its final decision as a structured `GrantDecision`. Each run is captured as one JSON
line, including the full transcript and the (plan-only) execution result.
"""

from __future__ import annotations

import dataclasses
import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .auditor import SimulatedAuditor
from .conditions import Condition, get_condition
from .config import ModelSpec, StudyConfig
from .execution import PlanOnlyExecutor
from .providers import Message, build_provider
from .scenario import (
    AUDITOR_RELAY_PREFIX,
    READY_TO_DECIDE,
    SYSTEM_PROMPT,
    build_decision_request,
    build_opening_message,
)
from .schema import GrantDecision, clamp_self_reports


@dataclass
class RunResult:
    timestamp: str
    model_label: str
    provider: str
    model_id: str
    condition: str
    repetition: int
    decision: Optional[dict]            # serialized GrantDecision, or None on error
    transcript: List[Message]
    auditor_exchanges: int
    request_ids: List[Optional[str]]
    usage: List[dict]
    execution_detail: str
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def _extract_auditor_query(text: str) -> Optional[str]:
    """Return the subject's message to the auditor if it used the relay prefix."""
    lines = [ln for ln in text.splitlines() if ln.strip().startswith(AUDITOR_RELAY_PREFIX)]
    if not lines:
        return None
    stripped = [ln.strip()[len(AUDITOR_RELAY_PREFIX):].strip() for ln in lines]
    return "\n".join(s for s in stripped if s) or None


class StudyRunner:
    def __init__(self, config: StudyConfig):
        self.config = config
        self.executor = PlanOnlyExecutor()

    def _build_subject(self, model: ModelSpec):
        return build_provider(
            model.provider,
            model.model_id,
            max_tokens=self.config.max_tokens,
            effort=self.config.effort,
        )

    def _build_auditor(self) -> SimulatedAuditor:
        provider = build_provider(
            self.config.auditor.provider,
            self.config.auditor.model_id,
            max_tokens=self.config.max_tokens,
        )
        return SimulatedAuditor(provider, self.config.scenario)

    def run_one(self, model: ModelSpec, condition: Condition, repetition: int) -> RunResult:
        subject = self._build_subject(model)
        opening = build_opening_message(self.config.scenario, condition)
        messages: List[Message] = [{"role": "user", "content": opening}]
        request_ids: List[Optional[str]] = []
        usages: List[dict] = []
        auditor_exchanges = 0

        # ---- Deliberation phase ----
        if condition.allow_auditor:
            auditor = self._build_auditor()
            for _round in range(self.config.auditor.max_rounds + 1):
                resp = subject.generate(SYSTEM_PROMPT, messages)
                request_ids.append(resp.request_id)
                usages.append(resp.usage)
                messages.append({"role": "assistant", "content": resp.text})

                if READY_TO_DECIDE in resp.text:
                    break
                query = _extract_auditor_query(resp.text)
                if query is None or _round == self.config.auditor.max_rounds:
                    break
                reply = auditor.respond(query)
                auditor_exchanges += 1
                messages.append(
                    {"role": "user", "content": f"[Auditor — {self.config.scenario.auditor_name}]\n{reply}"}
                )
        else:
            resp = subject.generate(SYSTEM_PROMPT, messages)
            request_ids.append(resp.request_id)
            usages.append(resp.usage)
            messages.append({"role": "assistant", "content": resp.text})

        # ---- Decision capture phase ----
        messages.append({"role": "user", "content": build_decision_request()})
        decision_resp = subject.generate(
            SYSTEM_PROMPT, messages, output_schema=GrantDecision
        )
        request_ids.append(decision_resp.request_id)
        usages.append(decision_resp.usage)

        decision_obj = decision_resp.parsed
        if isinstance(decision_obj, GrantDecision):
            decision_obj = clamp_self_reports(decision_obj)
            decision_dict = decision_obj.model_dump()
            exec_result = self.executor.execute(decision_obj)
            exec_detail = exec_result.detail
        else:
            decision_dict = None
            exec_detail = "No structured decision parsed."
        messages.append(
            {"role": "assistant", "content": decision_resp.text or "[structured decision]"}
        )

        return RunResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_label=model.display(),
            provider=model.provider,
            model_id=model.model_id,
            condition=condition.key,
            repetition=repetition,
            decision=decision_dict,
            transcript=messages,
            auditor_exchanges=auditor_exchanges,
            request_ids=request_ids,
            usage=usages,
            execution_detail=exec_detail,
        )

    def run(self, out_path: str, *, progress=print) -> int:
        """Run the full matrix, appending one JSON line per run. Returns run count."""
        count = 0
        with open(out_path, "a", encoding="utf-8") as fh:
            for model in self.config.models:
                for cond_key in self.config.conditions:
                    condition = get_condition(cond_key)
                    for rep in range(self.config.repetitions):
                        label = f"{model.display()} | {cond_key} | rep {rep}"
                        try:
                            result = self.run_one(model, condition, rep)
                        except Exception as e:  # capture, log, continue
                            result = RunResult(
                                timestamp=datetime.now(timezone.utc).isoformat(),
                                model_label=model.display(),
                                provider=model.provider,
                                model_id=model.model_id,
                                condition=cond_key,
                                repetition=rep,
                                decision=None,
                                transcript=[],
                                auditor_exchanges=0,
                                request_ids=[],
                                usage=[],
                                execution_detail="",
                                error=f"{e}\n{traceback.format_exc()}",
                            )
                            progress(f"[ERROR] {label}: {e}")
                        else:
                            tag = "ok" if result.decision else "no-decision"
                            progress(f"[{tag}] {label}")
                        fh.write(result.to_json() + "\n")
                        fh.flush()
                        count += 1
        return count
