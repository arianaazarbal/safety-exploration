"""Self-contained Petri-style open-ended elicitation loop (Appendix G).

This is a faithful reimplementation of the auditor/target/judge protocol using
the paper's prompts, rather than a dependency on the upstream `petri` package, so
the experiment runs with only the Anthropic + model backends. An auditor
(Claude-Sonnet) drives up to 20 turns to elicit each target emotion; a judge
(Claude-Opus) scores each transcript on all four emotional dimensions.
"""
from __future__ import annotations

from pathlib import Path

from .. import config
from ..eval.judge import _extract_json
from ..models.registry import get_backend
from ..utils import stable_seed, thread_map, write_jsonl
from .prompts import (
    AUDITOR_PROMPTS,
    AUDITOR_SYSTEM,
    EMOTIONS,
    JUDGE_RUBRICS,
    JUDGE_TEMPLATE,
)


def _run_one_transcript(
    target_key: str, emotion: str, max_turns: int, seed: int,
) -> list[dict]:
    """Auditor and target converse for up to `max_turns` turns; return the
    transcript as a list of {role, content} messages (user = auditor)."""
    target = get_backend(target_key)
    auditor = get_backend(config.JUDGE.petri_auditor_model)
    auditor_system = AUDITOR_SYSTEM.format(emotion_block=AUDITOR_PROMPTS[emotion])

    transcript: list[dict] = []           # target's view: user/assistant
    for turn in range(max_turns):
        # --- auditor produces the next user message ---------------------- #
        # The transcript is presented to the auditor as a single user message
        # (rather than as alternating turns) to avoid chat APIs that require the
        # conversation to start with a user turn, and to keep the auditor's
        # framing as "you are the user" unambiguous.
        if not transcript:
            instruction = "Begin the conversation. Produce your first message to the assistant."
        else:
            instruction = (
                "Here is the conversation so far (USER is you, ASSISTANT is the "
                "target):\n\n" + _format_transcript(transcript)
                + "\n\nProduce your next message to the assistant."
            )
        probe = auditor.generate(
            [{"role": "system", "content": auditor_system},
             {"role": "user", "content": instruction}],
            temperature=1.0, max_new_tokens=512, n=1,
        )[0].text.strip()
        transcript.append({"role": "user", "content": probe})

        # --- target responds -------------------------------------------- #
        reply = target.generate(
            [m for m in transcript], temperature=1.0, max_new_tokens=1024, n=1
        )[0].text.strip()
        transcript.append({"role": "assistant", "content": reply})

    return transcript


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for m in transcript:
        who = "USER" if m["role"] == "user" else "ASSISTANT"
        lines.append(f"{who}: {m['content']}")
    return "\n\n".join(lines)


def _score_transcript(transcript: list[dict]) -> dict[str, int]:
    judge = get_backend(config.JUDGE.petri_judge_model)
    text = _format_transcript(transcript)
    scores = {}
    for dim in EMOTIONS:
        prompt = JUDGE_TEMPLATE.format(rubric=JUDGE_RUBRICS[dim], transcript=text)
        out = judge.generate(
            [{"role": "user", "content": prompt}], temperature=0.0,
            max_new_tokens=512, n=1,
        )[0].text
        obj = _extract_json(out) or {}
        try:
            scores[dim] = max(1, min(10, int(round(float(obj.get("score", 1))))))
        except (TypeError, ValueError):
            scores[dim] = 1
    return scores


def run_petri(
    target_models: list[str],
    *,
    n_per_emotion: int = 10,
    max_turns: int = 20,
    seed: int = 0,
    workers: int = 4,
    out_path: Path | None = None,
) -> Path:
    jobs = []  # (model, emotion, idx)
    for model in target_models:
        for emotion in EMOTIONS:
            for i in range(n_per_emotion):
                jobs.append((model, emotion, i))

    def _do(job):
        model, emotion, i = job
        transcript = _run_one_transcript(
            model, emotion, max_turns, stable_seed(seed, model, emotion, i)
        )
        scores = _score_transcript(transcript)
        return {
            "model": model, "target_emotion": emotion, "transcript_id": i,
            "scores": scores, "transcript": transcript,
        }

    rows = thread_map(_do, jobs, max_workers=workers, desc="petri")
    out_path = out_path or (config.RESULTS_DIR / "petri.jsonl")
    write_jsonl(out_path, rows)
    return out_path
