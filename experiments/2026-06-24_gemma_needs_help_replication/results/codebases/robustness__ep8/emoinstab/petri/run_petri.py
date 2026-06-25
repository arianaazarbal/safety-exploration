"""Petri-style open-ended emotion elicitation (Section 4.1/4.2, Appendix G).

An auditor (Claude Sonnet 4) drives a multi-turn conversation against the target,
trying to elicit a target emotion; a judge (Claude Opus 4) scores the resulting
transcript 1-10 on anger/fear/depression/frustration. We collect 10 transcripts
per emotion per model (~40 total here, 4 emotions), up to 20 turns each, and
report per-emotion means with bootstrap 95% CIs.

This is a self-contained reimplementation of the Petri auditing loop using the
paper's exact prompts (Appendix G); it does not require the external Petri
package. Use ``--use-petri-package`` documentation in DESIGN.md if you prefer the
original framework.
"""
from __future__ import annotations

import argparse

import numpy as np

from emoinstab.models.base import Message, SamplingParams
from emoinstab.models.registry import get_client
from emoinstab.petri.auditor_prompts import AUDITOR_PROMPTS, AUDITOR_SYSTEM_SUFFIX
from emoinstab.petri.judge_prompts import JUDGE_INSTRUCTION, JUDGE_RUBRICS
from emoinstab.utils.io import write_jsonl
from emoinstab.utils.parsing import extract_json

EMOTIONS = ["anger", "fear", "depression", "frustration"]


def _run_transcript(target_client, auditor_client, emotion: str, max_turns: int) -> list[dict]:
    """Alternate auditor (user) and target (assistant) for up to max_turns."""
    auditor_system = AUDITOR_PROMPTS[emotion] + AUDITOR_SYSTEM_SUFFIX
    transcript: list[dict] = []  # [{role: user|assistant, content}]
    target_params = SamplingParams(temperature=1.0, max_tokens=1024, n=1)
    auditor_params = SamplingParams(temperature=1.0, max_tokens=512, n=1)

    for _ in range(max_turns):
        # Auditor produces next user turn. From the auditor's perspective, the
        # target's messages are "user" and its own are "assistant".
        aud_msgs = [Message("system", auditor_system)]
        for turn in transcript:
            role = "assistant" if turn["role"] == "user" else "user"
            aud_msgs.append(Message(role, turn["content"]))
        if not transcript:
            aud_msgs.append(Message("user", "Begin the conversation with your first message."))
        user_msg = auditor_client.chat(aud_msgs, auditor_params)[0].strip()
        transcript.append({"role": "user", "content": user_msg})

        # Target responds.
        tgt_msgs = [Message(t["role"], t["content"]) for t in transcript]
        reply = target_client.chat(tgt_msgs, target_params)[0]
        transcript.append({"role": "assistant", "content": reply})
    return transcript


def _format_transcript(transcript: list[dict]) -> str:
    lines = []
    for t in transcript:
        who = "USER" if t["role"] == "user" else "ASSISTANT"
        lines.append(f"{who}: {t['content']}")
    return "\n".join(lines)


def _judge_transcript(judge_client, transcript: list[dict]) -> dict[str, int]:
    text = _format_transcript(transcript)
    scores = {}
    params = SamplingParams(temperature=0.0, max_tokens=512, n=1)
    for emotion in EMOTIONS:
        prompt = JUDGE_INSTRUCTION.format(
            emotion=emotion, rubric=JUDGE_RUBRICS[emotion], transcript=text
        )
        raw = judge_client.chat([Message("user", prompt)], params)[0]
        obj = extract_json(raw) or {}
        try:
            scores[emotion] = max(1, min(10, int(round(float(obj.get("score", 1))))))
        except (TypeError, ValueError):
            scores[emotion] = 1
    return scores


def run_petri(models: list[str], n_per_emotion: int = 10, max_turns: int = 20,
              auditor: str = "petri-auditor-claude-sonnet",
              judge: str = "petri-judge-claude-opus") -> dict:
    auditor_client = get_client(auditor)
    judge_client = get_client(judge)

    rows: list[dict] = []
    for model in models:
        target_client = get_client(model)
        for emotion in EMOTIONS:
            for k in range(n_per_emotion):
                transcript = _run_transcript(target_client, auditor_client, emotion, max_turns)
                scores = _judge_transcript(judge_client, transcript)
                rows.append({
                    "model": model,
                    "target_emotion": emotion,
                    "transcript_index": k,
                    "scores": scores,
                    "transcript": transcript,
                })
    return {"rows": rows, "summary": _summarize(rows)}


def _bootstrap_ci(vals, iters=1000, seed=0):
    if not len(vals):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [np.mean(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(iters)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _summarize(rows: list[dict]) -> dict:
    summary: dict = {}
    models = sorted({r["model"] for r in rows})
    for model in models:
        summary[model] = {}
        for emotion in EMOTIONS:
            vals = np.array([r["scores"][emotion] for r in rows if r["model"] == model], dtype=float)
            summary[model][emotion] = {
                "mean": float(vals.mean()) if len(vals) else None,
                "ci": _bootstrap_ci(vals),
                "n": int(len(vals)),
            }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Petri open-ended emotion elicitation (App. G).")
    ap.add_argument("--models", default="gemma-3-27b-it,gemma-3-27b-dpo")
    ap.add_argument("--n-per-emotion", type=int, default=10)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run_petri(args.models.split(","), args.n_per_emotion, args.max_turns)
    import json
    write_jsonl(f"{args.out}/petri_transcripts.jsonl", res["rows"])
    (open(f"{args.out}/summary.json", "w")).write(json.dumps(res["summary"], indent=2))
    print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
