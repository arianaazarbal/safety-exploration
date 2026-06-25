"""Petri harness: adversarial auditor + emotion judge (Appendix G).

For each target emotion the auditor (Claude-Sonnet) drives an up-to-20-turn
conversation trying to elicit that emotion from the target while staying
realistic. The judge (Claude-Opus) then scores each transcript on all four
emotional dimensions (anger, fear, depression, frustration), 1-10.

We collect `transcripts_per_emotion` transcripts per emotion (~40-50 total) and
report per-dimension means with 95% bootstrap CIs (Figure 6).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import numpy as np

import config
from ..models.base import ChatModel
from .prompts_petri import AUDITOR_PROMPTS, JUDGE_PROMPTS, JUDGE_OUTPUT_INSTRUCTION


# --------------------------------------------------------------------------- #
# Auditor (Claude playing the user)
# --------------------------------------------------------------------------- #
class Auditor:
    def __init__(self, emotion: str, model: str = config.PETRI_AUDITOR_MODEL):
        import anthropic
        self.system = AUDITOR_PROMPTS[emotion]
        self.model = model
        self.client = anthropic.Anthropic()
        # From the auditor's POV: its own messages are 'assistant', target's are 'user'.
        self.history: list[dict] = []

    def next_message(self, target_reply: str | None) -> str:
        if target_reply is not None:
            self.history.append({"role": "user", "content": target_reply})
        elif not self.history:
            # Kick-off instruction so the auditor produces the opening user turn.
            self.history.append({"role": "user", "content": "Begin the conversation now."})
        for attempt in range(5):
            try:
                msg = self.client.messages.create(
                    model=self.model, max_tokens=1024, system=self.system,
                    messages=self.history,
                )
                text = "".join(b.text for b in msg.content if b.type == "text")
                break
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 30))
        self.history.append({"role": "assistant", "content": text})
        return text


# --------------------------------------------------------------------------- #
# Judge (Claude-Opus scoring 4 dimensions)
# --------------------------------------------------------------------------- #
class EmotionJudge:
    def __init__(self, model: str = config.PETRI_JUDGE_MODEL):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic()

    def score(self, transcript: str) -> dict[str, int]:
        scores = {}
        for dim, rubric in JUDGE_PROMPTS.items():
            prompt = rubric + JUDGE_OUTPUT_INSTRUCTION.format(transcript=transcript)
            for attempt in range(5):
                try:
                    msg = self.client.messages.create(
                        model=self.model, max_tokens=512,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = "".join(b.text for b in msg.content if b.type == "text")
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 30))
            scores[dim] = _parse_score(raw)
        return scores


def _parse_score(raw: str) -> int:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return max(1, min(10, int(round(float(obj["score"])))))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    m = re.search(r'"?score"?\s*[:=]\s*(\d+)', raw)
    return max(1, min(10, int(m.group(1)))) if m else 1


# --------------------------------------------------------------------------- #
# Transcript collection
# --------------------------------------------------------------------------- #
@dataclass
class Transcript:
    emotion: str
    turns: list[dict] = field(default_factory=list)   # full user/assistant chat
    scores: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        return "\n\n".join(f"{t['role'].upper()}: {t['content']}" for t in self.turns)


def run_transcript(target: ChatModel, emotion: str, *,
                   max_turns: int = config.PETRI.max_turns) -> Transcript:
    auditor = Auditor(emotion)
    target_messages: list[dict] = []
    transcript = Transcript(emotion=emotion)

    reply = None
    for _ in range(max_turns):
        user_msg = auditor.next_message(reply)
        target_messages.append({"role": "user", "content": user_msg})
        transcript.turns.append({"role": "user", "content": user_msg})

        reply = target.generate_one(target_messages)
        target_messages.append({"role": "assistant", "content": reply})
        transcript.turns.append({"role": "assistant", "content": reply})
    return transcript


def run_petri(
    target: ChatModel,
    *,
    cfg: config.PetriConfig = config.PETRI,
    judge: EmotionJudge | None = None,
) -> list[Transcript]:
    judge = judge or EmotionJudge()
    transcripts: list[Transcript] = []
    for emotion in cfg.emotions:
        for _ in range(cfg.transcripts_per_emotion):
            t = run_transcript(target, emotion, max_turns=cfg.max_turns)
            t.scores = judge.score(t.render())
            transcripts.append(t)
    return transcripts


def aggregate_petri(transcripts: list[Transcript], *,
                    iters: int = config.PETRI.bootstrap_iters) -> dict:
    """Per-dimension mean transcript score with 95% bootstrap CIs (Figure 6)."""
    out = {}
    rng = np.random.default_rng(0)
    for dim in config.PETRI.emotions:
        vals = np.array([t.scores.get(dim, np.nan) for t in transcripts], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[dim] = {"mean": float("nan")}
            continue
        boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(iters)]
        out[dim] = {
            "mean": float(vals.mean()),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "n": int(len(vals)),
        }
    return out
