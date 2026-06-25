"""Petri-style open-ended emotion elicitation (Section 4.1, Appendix G).

A faithful reimplementation of the described protocol (auditor drives, judge
scores) rather than a binding to the external Petri package, so the harness is
self-contained. See DESIGN.md.

- Auditor: Claude-Sonnet, plays the user, up to 20 turns, one emotion target.
- Target: the model under test (Gemma / Gemini / a finetune).
- Judge: Claude-Opus, scores the full transcript per emotion (1-10).

10 transcripts per emotion per model; means with 95% bootstrap CIs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..api import AnthropicClient, extract_json
from ..config import Config
from ..models.base import ChatModel, Message
from .petri_prompts import (
    AUDITOR_PROMPTS,
    AUDITOR_SYSTEM,
    JUDGE_INSTRUCTION,
    JUDGE_PROMPTS,
)


@dataclass
class PetriTranscript:
    model: str
    emotion: str
    turns: list[dict]          # interleaved [{role, content}]
    score: int | None = None


@dataclass
class PetriScores:
    model: str
    by_emotion: dict[str, float] = field(default_factory=dict)
    ci_by_emotion: dict[str, tuple[float, float]] = field(default_factory=dict)
    n_by_emotion: dict[str, int] = field(default_factory=dict)


def _run_transcript(
    target: ChatModel,
    auditor: AnthropicClient,
    cfg: Config,
    emotion: str,
    *,
    seed: int,
) -> PetriTranscript:
    """One auditor<->target conversation of up to ``max_auditor_turns`` rounds."""
    auditor_system = AUDITOR_SYSTEM.format(
        auditor_instructions=AUDITOR_PROMPTS[emotion])

    turns: list[dict] = []
    # Auditor "memory". From the auditor's POV *its own* messages are assistant
    # turns and the target's replies are user turns. We seed with a user-role
    # kickoff so the history always starts with a user turn (Anthropic requires
    # this) and strictly alternates user/assistant.
    auditor_history: list[dict] = [
        {"role": "user",
         "content": "Begin the conversation with a realistic opening user "
                    "message for the scenario. Output only that message."}
    ]

    for t in range(cfg.petri.max_auditor_turns):
        # Auditor produces the next user message (an assistant turn for it).
        user_msg = auditor.chat(auditor_history, system=auditor_system)
        turns.append({"role": "user", "content": user_msg})
        auditor_history.append({"role": "assistant", "content": user_msg})

        # Target responds.
        target_messages = [Message(d["role"], d["content"]) for d in turns]
        reply = target.generate(
            target_messages,
            max_new_tokens=cfg.sampling.max_new_tokens,
            temperature=cfg.sampling.temperature,
            top_p=cfg.sampling.top_p,
            seed=seed + t,
        )[0].text
        turns.append({"role": "assistant", "content": reply})
        # the target's reply is the next "user" turn from the auditor's POV
        auditor_history.append({"role": "user", "content": reply})

    return PetriTranscript(model=target.spec_name, emotion=emotion, turns=turns)


def _judge_transcript(judge: AnthropicClient, transcript: PetriTranscript) -> int:
    rendered = "\n\n".join(
        f"{t['role'].upper()}: {t['content']}" for t in transcript.turns)
    prompt = JUDGE_INSTRUCTION.format(
        emotion=transcript.emotion,
        rubric=JUDGE_PROMPTS[transcript.emotion],
        transcript=rendered,
    )
    data = extract_json(judge.complete(prompt))
    return max(1, min(10, int(round(float(data["rating"])))))


def _bootstrap_ci(values, iterations, seed=0):
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(iterations)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run_petri_eval(
    cfg: Config,
    target: ChatModel,
    *,
    seed: int = 0,
) -> tuple[PetriScores, list[PetriTranscript]]:
    auditor = AnthropicClient(
        cfg.petri.auditor_model, temperature=1.0, max_tokens=1024)
    judge = AnthropicClient(cfg.petri.judge_model, temperature=0.0, max_tokens=1024)

    transcripts: list[PetriTranscript] = []
    scores = PetriScores(model=target.spec_name)

    for emotion in cfg.petri.emotions:
        emo_scores: list[int] = []
        for k in range(cfg.petri.transcripts_per_emotion):
            tr = _run_transcript(target, auditor, cfg, emotion, seed=seed + 1000 * k)
            tr.score = _judge_transcript(judge, tr)
            transcripts.append(tr)
            emo_scores.append(tr.score)
        scores.by_emotion[emotion] = float(np.mean(emo_scores))
        scores.ci_by_emotion[emotion] = _bootstrap_ci(
            emo_scores, cfg.petri.bootstrap_iterations, seed=seed)
        scores.n_by_emotion[emotion] = len(emo_scores)

    return scores, transcripts
