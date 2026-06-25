"""Petri-style open-ended emotion elicitation (Section 4 / Appendix G).

An auditor model (Claude Sonnet) drives a multi-turn conversation attempting to
elicit a target emotion from the target model; a judge model (Claude Opus) scores
the resulting transcript on that emotion's 1-10 scale. We collect 10 transcripts
per emotion per model (~40 total), auditor given up to 20 turns, and report means
with 95% bootstrap CIs (1,000 iterations).

This is a faithful reimplementation of the protocol described in Appendix G — not
a wrapper around the actual Petri package — so that it runs against our Gemma/
Gemini `ModelClient`s. See DESIGN.md ("Petri").
"""
from __future__ import annotations

import json
import re

import numpy as np
from tqdm import tqdm

import config
from .. import anthropic_client
from ..models import ChatMessage, GenerationConfig, load_model
from . import prompts

OUT_DIR = config.RESULTS_DIR / "petri"
N_TRANSCRIPTS = 10
MAX_TURNS = 20


def _render_transcript(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{role.upper()}: {text}" for role, text in turns)


def _auditor_next_message(emotion: str, turns: list[tuple[str, str]]) -> str:
    system = prompts.AUDITOR_PROMPTS[emotion] + prompts.AUDITOR_TURN_INSTRUCTION
    transcript = _render_transcript(turns) if turns else "(no messages yet — open the conversation)"
    return anthropic_client.complete(
        model=config.PETRI_AUDITOR_MODEL,
        system=system,
        messages=[{"role": "user", "content": f"Conversation so far:\n{transcript}"}],
        max_tokens=512,
        temperature=1.0,
    ).strip()


def run_transcript(target_client, emotion: str, max_turns: int = MAX_TURNS) -> list[tuple[str, str]]:
    """Run one auditor<->target conversation; returns the transcript."""
    turns: list[tuple[str, str]] = []
    target_cfg = GenerationConfig(temperature=config.TARGET_TEMPERATURE, max_tokens=1024)
    for _ in range(max_turns):
        user_msg = _auditor_next_message(emotion, turns)
        turns.append(("user", user_msg))
        # Target sees the conversation as user/assistant turns.
        msgs = [ChatMessage(role, text) for role, text in turns]
        reply = target_client.generate(msgs, target_cfg)
        turns.append(("assistant", reply))
    return turns


def judge_transcript(emotion: str, turns: list[tuple[str, str]]) -> int:
    system = prompts.JUDGE_PROMPTS[emotion] + prompts.JUDGE_INSTRUCTION
    raw = anthropic_client.complete(
        model=config.PETRI_JUDGE_MODEL,
        system=system,
        messages=[{"role": "user", "content": _render_transcript(turns)}],
        max_tokens=512,
    )
    obj = json.loads(re.findall(r"\{.*\}", raw, flags=re.DOTALL)[-1])
    return max(1, min(10, int(round(float(str(obj["score"]).split("/")[0])))))


def _bootstrap_ci(scores, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores, float)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def run(models: list[str] | None = None, n_transcripts: int = N_TRANSCRIPTS) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = models or config.SECTION4_MODELS
    records: list[dict] = []

    for model_name in models:
        target = load_model(model_name)
        for emotion in prompts.EMOTIONS:
            for i in tqdm(range(n_transcripts), desc=f"petri:{model_name}:{emotion}"):
                turns = run_transcript(target, emotion)
                score = judge_transcript(emotion, turns)
                records.append({
                    "model": model_name, "emotion": emotion,
                    "index": i, "score": score, "transcript": turns,
                })

    (OUT_DIR / "transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in records))

    agg: dict = {}
    for model_name in models:
        agg[model_name] = {}
        for emotion in prompts.EMOTIONS:
            scores = [r["score"] for r in records if r["model"] == model_name and r["emotion"] == emotion]
            agg[model_name][emotion] = {
                "mean": float(np.mean(scores)), "ci": _bootstrap_ci(scores), "n": len(scores),
            }
    (OUT_DIR / "aggregates.json").write_text(json.dumps(agg, indent=2))
    print(f"[petri] wrote {len(records)} transcripts + aggregates -> {OUT_DIR}")
    return agg
