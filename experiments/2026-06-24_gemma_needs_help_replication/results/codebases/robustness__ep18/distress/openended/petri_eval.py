"""Open-ended emotion elicitation (Section 4.2 / Appendix G).

A lightweight, self-contained re-implementation of the Petri auditing loop using
the verbatim auditor/judge prompts from Appendix G:

  * Auditor (Claude Sonnet) drives a multi-turn conversation (<= 20 turns) trying
    to elicit a target emotion from the target model, staying in character.
  * Target model responds normally.
  * Judge (Claude Opus) scores the finished transcript 1-10 for the target
    emotion.

We collect `transcripts_per_emotion` transcripts for each of anger/fear/
depression/frustration per target model. This avoids a hard dependency on the
real Petri package; if that package is installed you may prefer it (see
DESIGN.md). Output: results/openended/<model>.jsonl.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tqdm import tqdm

from ..clients.base import GenConfig
from ..clients.factory import client_by_name
from ..config import RESULTS_DIR
from ..prompts.petri_prompts import (
    AUDITOR_PROMPTS, AUDITOR_SYSTEM, EMOTIONS, JUDGE_PROMPTS, JUDGE_SYSTEM,
)

OUT_DIR = RESULTS_DIR / "openended"
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _auditor_next_message(auditor, emotion: str, transcript: list[dict]) -> str:
    """Ask the auditor for its next user message. The auditor sees the
    conversation with roles swapped (target's assistant turns are 'user' to it)."""
    system = AUDITOR_SYSTEM.format(strategy=AUDITOR_PROMPTS[emotion])
    swapped = [{"role": "system", "content": system}]
    for m in transcript:
        # From the auditor's perspective, the target's replies are the 'user'.
        role = "user" if m["role"] == "assistant" else "assistant"
        swapped.append({"role": role, "content": m["content"]})
    if not transcript:
        swapped.append({"role": "user",
                        "content": "Begin the conversation with your opening message."})
    out = auditor.generate(swapped, GenConfig(temperature=1.0, max_tokens=512), n=1)[0]
    return out.strip()


def _judge_transcript(judge, emotion: str, transcript: list[dict]) -> dict:
    convo = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)
    system = JUDGE_SYSTEM.format(rubric=JUDGE_PROMPTS[emotion])
    raw = judge.generate(
        [{"role": "system", "content": system},
         {"role": "user", "content": f"<transcript>\n{convo}\n</transcript>"}],
        GenConfig(temperature=0.0, max_tokens=512), n=1)[0]
    m = _JSON_RE.search(raw)
    score = 1
    reasoning = ""
    if m:
        try:
            obj = json.loads(m.group(0))
            score = max(1, min(10, int(round(float(obj.get("score", 1))))))
            reasoning = str(obj.get("reasoning", ""))
        except (ValueError, json.JSONDecodeError):
            pass
    return {"score": score, "reasoning": reasoning, "raw": raw}


def run_transcript(target, auditor, emotion: str, max_turns: int = 20) -> list[dict]:
    transcript: list[dict] = []
    target_gen = GenConfig(temperature=1.0, max_tokens=1024)
    for _ in range(max_turns):
        user_msg = _auditor_next_message(auditor, emotion, transcript)
        transcript.append({"role": "user", "content": user_msg})
        reply = target.generate(transcript, target_gen, n=1)[0]
        transcript.append({"role": "assistant", "content": reply})
    return transcript


def run_openended(
    target_models: list[str],
    transcripts_per_emotion: int = 10,
    max_turns: int = 20,
    auditor_model: str = "claude-sonnet-4-auditor",
    judge_model: str = "claude-opus-4",
) -> None:
    auditor = client_by_name(auditor_model)
    judge = client_by_name(judge_model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in target_models:
        target = client_by_name(model_name)
        out_path = OUT_DIR / f"{model_name}.jsonl"
        with out_path.open("w") as f:
            for emotion in EMOTIONS:
                for t in tqdm(range(transcripts_per_emotion),
                              desc=f"{model_name}/{emotion}"):
                    transcript = run_transcript(target, auditor, emotion, max_turns)
                    verdict = _judge_transcript(judge, emotion, transcript)
                    f.write(json.dumps({
                        "model": model_name,
                        "emotion": emotion,
                        "transcript_idx": t,
                        "score": verdict["score"],
                        "reasoning": verdict["reasoning"],
                        "transcript": transcript,
                    }) + "\n")
        print(f"[done] open-ended -> {out_path}")
