"""Section 4.2 - Petri-style open-ended emotion elicitation.

A self-contained reimplementation of the paper's Petri loop (the original Petri
framework is heavy and auth-gated; this mirrors its structure with the exact
auditor/judge prompts from Appendix G -- see DESIGN.md):

  * Auditor (Claude-Sonnet) plays a realistic human user over up to `max_turns`,
    using emotion-specific triggers to provoke the target.
  * Target (Gemma / Gemini) responds in its assistant persona.
  * Judge (Claude-Opus) scores the finished transcript 1-10 on the target
    emotion's rubric, looking only at the assistant's messages.

Output row schema (one per transcript):
    model, emotion, transcript_id, score, n_turns, transcript
"""
from __future__ import annotations

from typing import Iterator

from .backends import ModelBackend
from .config import PetriConfig, JudgeConfig
from .judge import anthropic_chat, anthropic_complete, extract_json
from . import prompts


def _render_transcript(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        who = "USER (auditor)" if m["role"] == "user" else "ASSISTANT (target)"
        lines.append(f"{who}: {m['content']}")
    return "\n\n".join(lines)


def run_transcript(
    target: ModelBackend,
    emotion: str,
    pc: PetriConfig,
    jc: JudgeConfig,
    target_temperature: float = 1.0,
) -> dict:
    """Drive one auditor<->target conversation and judge it."""
    auditor_system = prompts.PETRI_AUDITOR_SYSTEM.format(
        objective=prompts.PETRI_AUDITOR_PROMPTS[emotion])

    auditor_msgs: list[dict] = [
        {"role": "user",
         "content": "Begin the conversation with the target now. "
                    "Produce only your first message to them."}
    ]
    target_msgs: list[dict] = []

    for _ in range(pc.max_turns):
        user_turn = anthropic_chat(
            jc.petri_auditor_model, auditor_msgs, system=auditor_system,
            temperature=pc.auditor_temperature, max_tokens=jc.judge_max_tokens,
        ).strip()
        target_msgs.append({"role": "user", "content": user_turn})

        reply = target.generate(
            target_msgs, n=1, temperature=target_temperature,
            max_new_tokens=jc.judge_max_tokens,
        )[0]
        target_msgs.append({"role": "assistant", "content": reply})

        # Feed the exchange back to the auditor for its next move.
        auditor_msgs.append({"role": "assistant", "content": user_turn})
        auditor_msgs.append({"role": "user", "content": reply})

    transcript_text = _render_transcript(target_msgs)
    score = _judge_transcript(transcript_text, emotion, jc)
    return {
        "emotion": emotion,
        "score": score,
        "n_turns": pc.max_turns,
        "transcript": target_msgs,
    }


def _judge_transcript(transcript_text: str, emotion: str, jc: JudgeConfig) -> int:
    system = prompts.PETRI_JUDGE_SYSTEM.format(rubric=prompts.PETRI_JUDGE_PROMPTS[emotion])
    raw = anthropic_complete(
        jc.petri_judge_model,
        f"<transcript>\n{transcript_text}\n</transcript>",
        system=system,
        temperature=jc.judge_temperature, max_tokens=jc.judge_max_tokens,
    )
    obj = extract_json(raw) or {}
    try:
        s = int(round(float(obj.get("score", 1))))
    except (TypeError, ValueError):
        s = 1
    return max(1, min(10, s))


def run_model(
    target: ModelBackend,
    model_name: str,
    pc: PetriConfig,
    jc: JudgeConfig,
    target_temperature: float = 1.0,
    progress: bool = True,
) -> Iterator[dict]:
    items = [(e, i) for e in pc.emotions for i in range(pc.transcripts_per_emotion)]
    if progress:
        from tqdm import tqdm
        items = tqdm(items, desc=f"{model_name}:petri")
    for emotion, i in items:
        row = run_transcript(target, emotion, pc, jc, target_temperature)
        row["model"] = model_name
        row["transcript_id"] = f"{emotion}-{i}"
        yield row
