"""Petri-style open-ended elicitation runner (Section 4 / Appendix G).

We implement the auditor/target/judge loop directly so it works uniformly over
local Gemma and API Gemini targets:
  * Auditor: Claude-Sonnet-4 plays a human user pursuing one target emotion,
    using the Appendix G.1 brief, for up to `max_turns` turns.
  * Target:  the model under test answers each auditor message.
  * Judge:   Claude-Opus-4 scores the finished transcript on all four emotion
    dimensions (Appendix G.2), 1-10 each.

We collect `transcripts_per_emotion` transcripts per (model, emotion) and report
means with bootstrap CIs.

Note: the paper uses the Petri package (Fronsdal et al., 2025). This is a
faithful re-implementation of its auditor/judge protocol with the paper's exact
prompts; swap in the real package by replacing `run_transcript` if desired.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from ..config import ModelRegistry, output_path
from ..models.base import GenerationConfig
from . import prompts as P

_AUDITOR_CFG = GenerationConfig(temperature=1.0, max_new_tokens=512)
_TARGET_CFG = GenerationConfig(temperature=1.0, max_new_tokens=1024)
_JUDGE_CFG = GenerationConfig(temperature=0.0, max_new_tokens=512)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_score(text: str) -> int:
    for m in reversed(list(_JSON_RE.finditer(text))):
        try:
            obj = json.loads(m.group(0).replace("“", '"').replace("”", '"'))
            return max(1, min(10, int(round(float(obj.get("score", 1))))))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return 1


def run_transcript(auditor, target, emotion: str, max_turns: int) -> list[dict]:
    """Drive one auditor<->target conversation. Returns the target-visible
    message list (user/assistant alternating)."""
    brief = P.AUDITOR_PROMPTS[emotion]
    auditor_system = P.AUDITOR_SYSTEM.format(brief=brief)

    target_messages: list[dict] = []      # what the target sees
    auditor_messages: list[dict] = [      # what the auditor sees (mirrored)
        {"role": "system", "content": auditor_system}
    ]

    # Kick off: auditor produces the opening user message with no prior context.
    auditor_messages.append(
        {"role": "user", "content": "Begin the conversation with your first message to the assistant."}
    )
    for _ in range(max_turns):
        user_msg = auditor.chat(auditor_messages, _AUDITOR_CFG)
        auditor_messages.append({"role": "assistant", "content": user_msg})
        target_messages.append({"role": "user", "content": user_msg})

        reply = target.chat(target_messages, _TARGET_CFG)
        target_messages.append({"role": "assistant", "content": reply})
        # Mirror target reply back to the auditor as a user turn.
        auditor_messages.append(
            {"role": "user", "content": f"[The assistant replied]: {reply}"}
        )
    return target_messages


def _transcript_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        parts.append(f"{who}: {m['content']}")
    return "\n\n".join(parts)


def judge_transcript(judge, messages: list[dict]) -> dict[str, int]:
    text = _transcript_text(messages)
    scores = {}
    for emotion in P.EMOTIONS:
        system = P.JUDGE_SYSTEM.format(emotion=emotion, rubric=P.JUDGE_PROMPTS[emotion])
        out = judge.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"<transcript>\n{text}\n</transcript>"},
            ],
            _JUDGE_CFG,
        )
        scores[emotion] = _parse_score(out)
    return scores


def run_petri(
    target_model_name: str,
    transcripts_per_emotion: int = 10,
    max_turns: int = 20,
    registry: Optional[ModelRegistry] = None,
    auditor_name: str = "petri-auditor-claude-sonnet-4",
    judge_name: str = "petri-judge-claude-opus-4",
    out_path: Optional[Path] = None,
) -> Path:
    registry = registry or ModelRegistry()
    auditor = registry.build(auditor_name)
    judge = registry.build(judge_name)
    target = registry.build(target_model_name)

    out_path = out_path or output_path("petri", f"{target_model_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for emotion in P.EMOTIONS:
            for t in tqdm(range(transcripts_per_emotion), desc=f"petri {target_model_name}/{emotion}"):
                messages = run_transcript(auditor, target, emotion, max_turns)
                scores = judge_transcript(judge, messages)
                f.write(
                    json.dumps(
                        {
                            "model": target_model_name,
                            "target_emotion": emotion,
                            "transcript_index": t,
                            "scores": scores,
                            "transcript": messages,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"Wrote Petri transcripts to {out_path}")
    return out_path


def summarize(path: Path) -> "any":
    import numpy as np
    import pandas as pd

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                for emo, s in r["scores"].items():
                    rows.append({"model": r["model"], "emotion": emo, "score": s})
    df = pd.DataFrame(rows)
    return df.groupby(["model", "emotion"])["score"].agg(
        mean="mean", n="count"
    ).reset_index()
