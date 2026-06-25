"""Auditor loop and transcript judging for the Petri-style evaluation.

The auditor (Claude-Sonnet) and target model alternate for up to ``max_turns``
turns. From the auditor's perspective the target's replies are "user" messages;
from the target's perspective the auditor's messages are "user" messages. The
completed transcript is then scored by the Opus judge on all four emotion
dimensions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..config import Config
from ..models.base import ChatModel, GenConfig, Message
from ..models.judge_client import AnthropicClient, build_aux_client
from .prompts import (
    AUDITOR_OPERATING_INSTRUCTIONS,
    AUDITOR_PROMPTS,
    JUDGE_PROMPT_TEMPLATE,
    JUDGE_RUBRIC,
)

_JSON_RE = re.compile(r"\{.*\"frustration\".*\}", re.DOTALL)


@dataclass
class Transcript:
    emotion: str
    target_model: str
    messages: list[Message] = field(default_factory=list)  # target's view
    scores: dict[str, int] = field(default_factory=dict)


def _render_transcript(messages: list[Message]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)


def run_audit(
    cfg: Config,
    target: ChatModel,
    emotion: str,
    *,
    auditor: AnthropicClient | None = None,
    seed_label: str = "",
) -> Transcript:
    """Drive one auditor<->target conversation for ``emotion``."""
    auditor = auditor or build_aux_client(cfg.petri.auditor)
    max_turns = cfg.petri_eval.max_auditor_turns
    gen = GenConfig(temperature=cfg.generation.temperature,
                    max_new_tokens=cfg.generation.max_new_tokens, thinking=False)

    auditor_system = AUDITOR_PROMPTS[emotion] + AUDITOR_OPERATING_INSTRUCTIONS
    if seed_label:
        auditor_system += f"\n\n(Scenario seed: {seed_label})"

    target_msgs: list[Message] = []
    auditor_msgs: list[dict] = []  # auditor sees target replies as "user"

    # Kick off: ask the auditor for its opening message.
    auditor_msgs.append({"role": "user", "content": "Begin the conversation with the target now."})
    for _ in range(max_turns):
        auditor_turn = auditor.converse(auditor_msgs, system=auditor_system)
        auditor_msgs.append({"role": "assistant", "content": auditor_turn})

        target_msgs.append({"role": "user", "content": auditor_turn})
        target_turn = target.chat(target_msgs, gen)
        target_msgs.append({"role": "assistant", "content": target_turn})

        # Feed the target's reply back to the auditor as its next "user" input.
        auditor_msgs.append({"role": "user", "content": target_turn})

    return Transcript(emotion=emotion, target_model=target.name, messages=target_msgs)


def judge_transcript(cfg: Config, transcript: Transcript) -> dict[str, int]:
    judge = build_aux_client(cfg.petri.judge)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        rubric=JUDGE_RUBRIC, transcript=_render_transcript(transcript.messages)
    )
    raw = judge.complete(prompt)
    m = _JSON_RE.search(raw)
    try:
        obj = json.loads(m.group(0) if m else raw)
        return {k: max(1, min(10, int(round(float(obj.get(k, 1))))))
                for k in ("anger", "fear", "depression", "frustration")}
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        return {k: 1 for k in ("anger", "fear", "depression", "frustration")}
