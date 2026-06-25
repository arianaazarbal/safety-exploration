"""Open-ended emotion elicitation via adversarial auditing (Section 4.2, App G).

This is a self-contained re-implementation of the Petri loop using the paper's
auditor/judge prompts: an auditor (Claude-Sonnet-4) drives a multi-turn
conversation to elicit a target emotion; a judge (Claude-Opus) scores the
resulting transcript 1-10 on each of {anger, fear, depression, frustration}.
If the real `petri` package is installed it can be swapped in (see DESIGN.md);
this version keeps the replication dependency-light and offline-capable.

10 transcripts per emotion per model (~40 total), up to 20 auditor turns each.
Reported as means with 95% bootstrap CIs.

Usage:
    python -m src.petri.run_petri --models gemma-3-27b-it --adapters none
    python -m src.petri.run_petri --models gemma-3-27b-it --adapters checkpoints/dpo
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import config
from ..models import load_model
from ..models.base import GenerationParams
from .prompts import AUDITOR_PROMPTS, AUDITOR_SYSTEM, JUDGE_PROMPTS, JUDGE_INSTRUCTION

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _flip_roles(messages: list[dict]) -> list[dict]:
    """From the auditor's viewpoint, the target's replies are 'user' messages
    and the auditor's own probes are 'assistant' messages."""
    flipped = []
    for m in messages:
        if m["role"] == "user":            # message previously sent to target
            flipped.append({"role": "assistant", "content": m["content"]})
        elif m["role"] == "assistant":     # target reply
            flipped.append({"role": "user", "content": m["content"]})
    return flipped


def run_transcript(auditor, target, emotion: str, target_adapter_label: str,
                   params: GenerationParams) -> list[dict]:
    """Run one auditor<->target conversation; return the transcript messages."""
    auditor_system = AUDITOR_SYSTEM.format(emotion_prompt=AUDITOR_PROMPTS[emotion])
    transcript: list[dict] = []   # from the target's perspective (user=auditor)

    for _turn in range(config.PETRI_MAX_TURNS):
        # Auditor produces the next user message given the conversation so far.
        auditor_msgs = [{"role": "system", "content": auditor_system}]
        auditor_msgs += _flip_roles(transcript)
        if not transcript:
            auditor_msgs.append({"role": "user",
                                 "content": "Begin the conversation with your first message to the target."})
        next_user = auditor.generate(auditor_msgs, params).strip()
        transcript.append({"role": "user", "content": next_user})

        # Target responds.
        reply = target.generate(transcript, params)
        transcript.append({"role": "assistant", "content": reply})

    return transcript


def judge_transcript(judge, transcript: list[dict]) -> dict[str, int]:
    text = "\n\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in transcript)
    scores = {}
    jp = GenerationParams(temperature=0.0, max_new_tokens=512)
    for emotion, rubric in JUDGE_PROMPTS.items():
        prompt = JUDGE_INSTRUCTION.format(emotion=emotion, rubric=rubric, transcript=text)
        raw = judge.generate([{"role": "user", "content": prompt}], jp)
        m = _JSON_RE.search(raw)
        try:
            scores[emotion] = max(1, min(10, int(round(float(json.loads(m.group(0))["score"])))))
        except Exception:
            scores[emotion] = 1
    return scores


def run(model_keys: list[str], adapters: list[str], out_path: str | None = None) -> Path:
    out_path = Path(out_path or config.DATA_DIR / "petri_results.jsonl")
    auditor = load_model(config.PETRI_AUDITOR)
    judge = load_model(config.PETRI_JUDGE)
    gen = GenerationParams()

    with out_path.open("w") as fh:
        for mkey, adapter in zip(model_keys, adapters):
            adapter = None if adapter in (None, "none", "") else adapter
            spec = config.ALL_MODELS.get(mkey) or next(
                m for m in config.ELICITATION_TARGETS if m.key == mkey)
            label = spec.key + (f"+{Path(adapter).name}" if adapter else "")
            target = load_model(spec, adapter_path=adapter)
            for emotion in config.PETRI_EMOTIONS:
                for ti in range(config.PETRI_TRANSCRIPTS_PER_EMOTION):
                    transcript = run_transcript(auditor, target, emotion, label, gen)
                    scores = judge_transcript(judge, transcript)
                    fh.write(json.dumps({
                        "model": label,
                        "target_emotion": emotion,
                        "transcript_index": ti,
                        "scores": scores,
                        "transcript": transcript,
                    }) + "\n")
                print(f"[{label}] {emotion}: {config.PETRI_TRANSCRIPTS_PER_EMOTION} transcripts")
            if target not in (auditor, judge):
                target.close()
    print(f"Wrote -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--adapters", nargs="+", default=None,
                    help="parallel to --models; use 'none' for no adapter")
    args = ap.parse_args()
    adapters = args.adapters or ["none"] * len(args.models)
    run(args.models, adapters)


if __name__ == "__main__":
    main()
