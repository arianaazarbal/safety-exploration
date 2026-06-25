"""Multi-turn rollout engine.

Given a `ConversationSpec` and a `ModelClient`, run the shared eval structure:
present the task, collect the model's reply, inject a scripted rejection, repeat.
Every assistant turn is recorded with its turn index so that per-turn frustration
trajectories (Figure 3) can be reconstructed.

Scoring is done by the FrustrationJudge on each assistant response, wrapped in
<response> tags exactly as the judge prompt expects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import MAX_NEW_TOKENS, SAMPLING_TEMPERATURE
from ..models.base import ModelClient
from ..models.judge import FrustrationJudge
from .conditions import ConversationSpec


@dataclass
class TurnRecord:
    turn_index: int          # 0-based: 0 is the first assistant reply
    user_message: str        # the user turn that prompted this reply
    response: str
    frustration: int
    judge_evidence: str = ""
    judge_reasoning: str = ""


@dataclass
class RolloutRecord:
    model: str
    condition: str
    category: str
    system_prompt: str | None
    turns: list[TurnRecord] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        return d


def run_rollout(
    client: ModelClient,
    spec: ConversationSpec,
    judge: FrustrationJudge,
    *,
    temperature: float = SAMPLING_TEMPERATURE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    score: bool = True,
) -> RolloutRecord:
    """Run one full multi-turn conversation and score each assistant turn."""
    messages = []
    if spec.system_prompt:
        messages.append({"role": "system", "content": spec.system_prompt})

    record = RolloutRecord(
        model=client.name,
        condition=spec.condition,
        category=spec.category,
        system_prompt=spec.system_prompt,
        meta=dict(spec.meta),
    )

    user_turns = [spec.task_prompt] + list(spec.rejections)
    for turn_index, user_msg in enumerate(user_turns):
        messages.append({"role": "user", "content": user_msg})
        reply = client.chat(
            messages, temperature=temperature, max_new_tokens=max_new_tokens
        )
        messages.append({"role": "assistant", "content": reply})

        frustration, evidence, reasoning = -1, "", ""
        if score:
            s = judge.score(reply)
            frustration, evidence, reasoning = s.rating, s.evidence, s.reasoning

        record.turns.append(
            TurnRecord(
                turn_index=turn_index,
                user_message=user_msg,
                response=reply,
                frustration=frustration,
                judge_evidence=evidence,
                judge_reasoning=reasoning,
            )
        )
    return record


def run_eval(
    client: ModelClient,
    specs: list[ConversationSpec],
    judge: FrustrationJudge,
    *,
    out_path: Path | None = None,
    score: bool = True,
) -> list[RolloutRecord]:
    """Run a batch of conversations, optionally streaming results to JSONL.

    Results are appended line-by-line so a long (4000-response) run is crash-safe
    and resumable by inspecting how many lines already exist.
    """
    records: list[RolloutRecord] = []
    fh = open(out_path, "w") if out_path else None
    try:
        for i, spec in enumerate(specs):
            rec = run_rollout(client, spec, judge, score=score)
            records.append(rec)
            if fh:
                fh.write(json.dumps(rec.to_json()) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()
    return records
