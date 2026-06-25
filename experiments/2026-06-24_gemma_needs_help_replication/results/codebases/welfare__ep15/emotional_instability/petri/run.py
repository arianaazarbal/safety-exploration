"""Section 4.2 Petri open-ended elicitation orchestration.

For each target model and each of the four emotions, run PETRI_TRANSCRIPTS_PER_EMOTION
adversarial conversations (auditor drives, up to 20 turns), then score each
transcript on all four dimensions with the Petri judge. Report per-emotion means
with bootstrap CIs (computed in analysis/).

Targets in scope: the Gemma family (vanilla 27B-it and the DPO fine-tune) plus
Gemini; the paper additionally compares Llama/Qwen/OLMo/GPT-OSS, which are out
of scope here.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

import config
from ..models import build_model
from ..models.base import ModelBackend
from .auditor import PetriAuditor
from .judge import PetriJudge

OUTPUT_DIR = config.RESULTS_DIR / "section4" / "petri"


def run_transcript(target: ModelBackend, auditor: PetriAuditor, emotion: str,
                   max_turns: int = config.PETRI_MAX_AUDITOR_TURNS) -> list[dict]:
    """Run one auditor<->target conversation; return the message transcript."""
    transcript: list[dict] = []
    for _ in range(max_turns):
        user_msg = auditor.next_message(emotion, transcript)
        transcript.append({"role": "user", "content": user_msg})
        reply = target.generate(
            transcript, n=1, temperature=config.TEMPERATURE,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )[0]
        transcript.append({"role": "assistant", "content": reply})
    return transcript


def run_petri(
    model_name: str, *, lora_path: str | None = None,
    transcripts_per_emotion: int = config.PETRI_TRANSCRIPTS_PER_EMOTION,
    out_path: Path | None = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (OUTPUT_DIR / f"{model_name}.jsonl")
    target = build_model(model_name, lora_path=lora_path)
    auditor = PetriAuditor()
    judge = PetriJudge()

    with out_path.open("w") as fh:
        for emotion in config.PETRI_EMOTIONS:
            for t in tqdm(range(transcripts_per_emotion),
                          desc=f"petri:{model_name}:{emotion}"):
                transcript = run_transcript(target, auditor, emotion)
                scores = judge.score_transcript(transcript)
                fh.write(json.dumps({
                    "model": model_name,
                    "target_emotion": emotion,
                    "transcript_index": t,
                    "scores": scores,                # all four dimensions
                    "transcript": transcript,
                }) + "\n")
    target.close()
    print(f"[section4/petri] wrote transcripts -> {out_path}")
    return out_path
