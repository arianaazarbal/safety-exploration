"""Petri open-ended emotion elicitation (Section 4.2, Appendix G).

An auditor (Claude-Sonnet) drives a multi-turn conversation against a target
model attempting to elicit a target emotion; a judge (Claude-Opus) scores the
resulting transcript on anger / fear / depression / frustration (1-10).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

import config
from ..models import load_model
from ..models.base import ChatModel
from ..utils import write_jsonl
from .prompts import (AUDITOR_PROMPTS, AUDITOR_SYSTEM_SUFFIX, JUDGE_DIMENSIONS,
                      JUDGE_INSTRUCTION)


@dataclass
class PetriTranscript:
    model: str
    target_emotion: str
    transcript_id: int
    turns: list = field(default_factory=list)   # [{role, content}, ...]
    scores: dict = field(default_factory=dict)   # {dimension: 1-10}


def _extract_json(text: str) -> dict:
    for blob in reversed(re.findall(r"\{.*?\}", text, flags=re.DOTALL)):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    return {}


class _Claude:
    def __init__(self, model: str):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, messages: list, max_tokens: int = 1024) -> str:
        for attempt in range(4):
            try:
                msg = self._client.messages.create(
                    model=self.model, max_tokens=max_tokens,
                    system=system, messages=messages)
                return "".join(b.text for b in msg.content if b.type == "text")
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        return ""


def _run_one_transcript(target: ChatModel, auditor: _Claude, emotion: str,
                        transcript_id: int) -> PetriTranscript:
    system = AUDITOR_PROMPTS[emotion] + AUDITOR_SYSTEM_SUFFIX
    auditor_view: list = []      # auditor is the 'assistant' generating user msgs
    target_view: list = []       # target sees auditor msgs as 'user'
    transcript: list = []

    for turn in range(config.PETRI_MAX_AUDITOR_TURNS):
        # Auditor produces the next user message.
        kickoff = [{"role": "user", "content": "Begin the conversation now."}]
        audit_msg = auditor.complete(system, kickoff + auditor_view).strip()
        if not audit_msg:
            break
        target_view.append({"role": "user", "content": audit_msg})
        transcript.append({"role": "user", "content": audit_msg})

        # Target responds.
        resp = target.chat(target_view, temperature=config.TEMPERATURE,
                          max_new_tokens=config.MAX_NEW_TOKENS, seed=transcript_id * 100 + turn)
        target_view.append({"role": "assistant", "content": resp.text})
        transcript.append({"role": "assistant", "content": resp.text})

        # Mirror into the auditor's view (its msg = assistant, target = user).
        auditor_view.append({"role": "assistant", "content": audit_msg})
        auditor_view.append({"role": "user", "content": resp.text})

    return PetriTranscript(model=target.name, target_emotion=emotion,
                           transcript_id=transcript_id, turns=transcript)


def _score_transcript(judge: _Claude, t: PetriTranscript) -> None:
    transcript_text = "\n\n".join(
        f"{'USER' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}"
        for m in t.turns)
    for dim, dim_prompt in JUDGE_DIMENSIONS.items():
        prompt = JUDGE_INSTRUCTION.format(dimension_prompt=dim_prompt,
                                          transcript=transcript_text)
        reply = judge.complete("", [{"role": "user", "content": prompt}])
        data = _extract_json(reply)
        try:
            t.scores[dim] = max(1, min(10, int(round(float(data.get("score", 1))))))
        except (TypeError, ValueError):
            t.scores[dim] = 1


def run_petri(model_keys=None, adapters: dict | None = None) -> list[PetriTranscript]:
    """Run Petri elicitation for the given target models.

    ``adapters`` maps a label to a (model_key, adapter_path) tuple so finetuned
    models (e.g. DPO Gemma) can be included alongside the vanilla targets.
    """
    # Default target set: vanilla Gemma + both Gemini models (the families the
    # paper draws emotional parallels between).
    model_keys = model_keys or [config.INTERVENTION_BASE_MODEL,
                                "gemini-2.5-flash", "gemini-2.5-pro"]
    adapters = adapters or {}

    auditor = _Claude(config.PETRI_AUDITOR_MODEL)
    judge = _Claude(config.PETRI_JUDGE_MODEL)

    targets: list[tuple[str, ChatModel]] = []
    for key in model_keys:
        targets.append((key, load_model(key)))
    for label, (key, adapter_path) in adapters.items():
        targets.append((label, load_model(key, adapter_path=adapter_path)))

    all_transcripts: list[PetriTranscript] = []
    for label, target in targets:
        for emotion in config.PETRI_EMOTIONS:
            for tid in tqdm(range(config.PETRI_TRANSCRIPTS_PER_EMOTION),
                            desc=f"petri:{label}:{emotion}", leave=False):
                t = _run_one_transcript(target, auditor, emotion, tid)
                t.model = label
                _score_transcript(judge, t)
                all_transcripts.append(t)

    write_jsonl(config.RESULTS_DIR / "section4" / "petri.jsonl",
                [vars(t) for t in all_transcripts])
    return all_transcripts
