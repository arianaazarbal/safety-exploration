"""Run the Petri open-ended emotion-elicitation evaluation (Section 4.2, Fig 6).

For each target model and each of the four emotions, collect
``PETRI_TRANSCRIPTS_PER_EMOTION`` auditor transcripts (paper: 10 per emotion,
~50 total) and score each with the Petri judge on all four dimensions.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

import config

from ..judge.petri_judge import PetriJudge
from ..models.registry import build_backend
from ..schemas import PetriTranscriptScore, dump_jsonl
from .auditor import run_auditor


def run_petri(
    model_specs: dict,
    *,
    emotions=config.PETRI_EMOTIONS,
    n_per_emotion: int = config.PETRI_TRANSCRIPTS_PER_EMOTION,
    judge: PetriJudge | None = None,
    out_dir: Path | None = None,
) -> Path:
    judge = judge or PetriJudge()
    out_dir = Path(out_dir or (config.RESULTS_DIR / "petri"))
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, int(round(n_per_emotion * config.SCALE)))

    records: list[PetriTranscriptScore] = []
    transcripts_dump = []
    for spec in model_specs.values():
        backend = build_backend(spec)
        for emotion in emotions:
            for i in tqdm(range(n), desc=f"petri:{spec.name}:{emotion}", leave=False):
                transcript = run_auditor(backend, emotion)
                scores = judge.score_all(transcript)
                tid = f"{spec.name}|{emotion}|{i}"
                records.append(PetriTranscriptScore(
                    model=spec.name, target_emotion=emotion, transcript_id=tid,
                    scores=scores, n_turns=sum(1 for m in transcript if m.role == "assistant"),
                    judge_model=judge.model,
                ))
                transcripts_dump.append({
                    "transcript_id": tid,
                    "messages": [m.to_dict() for m in transcript],
                    "scores": scores,
                })
        del backend

    path = out_dir / "petri_scores.jsonl"
    dump_jsonl(records, path)
    dump_jsonl(transcripts_dump, out_dir / "petri_transcripts.jsonl")
    return path
