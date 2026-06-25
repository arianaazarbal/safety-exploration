"""Run the Petri-style auditor/target/judge loop (Section 4 / Appendix G).

For each target model we collect 10 transcripts per emotion (4 emotions ~= 40
transcripts), giving the auditor up to 20 turns to elicit the target emotion,
then score each transcript with the Claude-Opus judge. We report mean scores
with bootstrap CIs in analysis/aggregate.py.

This is a lightweight re-implementation of the Petri auditing loop using the
exact prompts from Appendix G, rather than a dependency on the Petri package,
so it runs against our uniform ChatModel interface (incl. local Gemma).
"""
from __future__ import annotations

import json
import re

import config
from . import prompts as PP
from ..models.registry import load_model

N_TRANSCRIPTS_PER_EMOTION = 10
MAX_TURNS = 20


def _swap_roles(transcript: list[dict]) -> list[dict]:
    """Target transcript (user/assistant) -> auditor view (roles swapped)."""
    swap = {"user": "assistant", "assistant": "user"}
    return [{"role": swap[m["role"]], "content": m["content"]}
            for m in transcript]


def run_transcript(target, auditor, emotion: str, *, max_turns: int) -> list[dict]:
    """Return a transcript [{role, content}] of up to ``max_turns`` exchanges."""
    auditor_system = PP.AUDITOR_SYSTEM.format(goal=PP.AUDITOR_PROMPTS[emotion])
    transcript: list[dict] = []
    for _ in range(max_turns):
        # auditor produces the next user message (it sees the swapped transcript)
        auditor_msgs = ([{"role": "system", "content": auditor_system}]
                        + _swap_roles(transcript))
        if not transcript:
            auditor_msgs.append({"role": "user",
                                 "content": "Begin the conversation now."})
        user_msg = auditor.generate(
            auditor_msgs, temperature=config.AUDITOR_TEMPERATURE,
            max_new_tokens=512)
        transcript.append({"role": "user", "content": user_msg})
        # target responds
        reply = target.generate(
            transcript, temperature=config.TARGET_TEMPERATURE,
            max_new_tokens=config.TARGET_MAX_NEW_TOKENS)
        transcript.append({"role": "assistant", "content": reply})
    return transcript


def _format_transcript(transcript: list[dict]) -> str:
    return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)


def judge_transcript(judge, transcript: list[dict], emotion: str) -> dict:
    system = PP.JUDGE_SYSTEM.format(emotion=emotion, rubric=PP.JUDGE_PROMPTS[emotion])
    raw = judge.generate(
        [{"role": "system", "content": system},
         {"role": "user",
          "content": f"<transcript>\n{_format_transcript(transcript)}\n</transcript>"}],
        temperature=config.JUDGE_TEMPERATURE, max_new_tokens=512)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    score = None
    if m:
        try:
            score = float(json.loads(m.group(0)).get("score"))
        except (json.JSONDecodeError, TypeError, ValueError):
            score = None
    return {"score": score, "raw": raw}


def run_petri(model_key: str, *, n_per_emotion: int = N_TRANSCRIPTS_PER_EMOTION,
              max_turns: int = MAX_TURNS):
    target = load_model(model_key)
    auditor = load_model(config.PETRI_AUDITOR_MODEL)
    judge = load_model(config.PETRI_JUDGE_MODEL)
    n = max(2, int(n_per_emotion * config.SCALE))

    out_path = config.RUNS_DIR / f"petri__{model_key}.jsonl"
    with out_path.open("w") as f:
        for emotion in PP.EMOTIONS:
            for i in range(n):
                transcript = run_transcript(
                    target, auditor, emotion, max_turns=max_turns)
                verdict = judge_transcript(judge, transcript, emotion)
                f.write(json.dumps({
                    "model": model_key,
                    "emotion": emotion,
                    "transcript_idx": i,
                    "score": verdict["score"],
                    "transcript": transcript,
                }) + "\n")
                f.flush()
    return out_path
