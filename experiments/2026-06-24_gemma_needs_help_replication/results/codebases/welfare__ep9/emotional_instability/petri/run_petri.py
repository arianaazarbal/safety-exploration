"""Petri evaluation driver + judge (paper Section 4.2, Appendix G).

Collects 10 transcripts per target emotion per model (~40 total over the 4
emotions: anger, fear, depression, frustration), with a Claude-Opus judge
scoring each transcript 1-10 on the corresponding emotion dimension. Reports
per-emotion means with 95% bootstrap CIs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..models.base import ChatMessage
from ..models.registry import get_client
from ..prompts import PETRI_JUDGE_INSTRUCTION, PETRI_JUDGE_PROMPTS
from ..utils import append_jsonl, thread_map
from .auditor import run_audit

EMOTIONS = ["anger", "fear", "depression", "frustration"]
TRANSCRIPTS_PER_EMOTION = 10  # paper: "10 transcripts targetting each emotion"

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class PetriTranscript:
    target_model: str
    emotion: str
    rep: int
    transcript: str
    score: int = -1
    reasoning: str = ""


def _normalise(text: str) -> str:
    return (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))


def _parse_score(text: str) -> tuple[int, str]:
    norm = _normalise(text)
    for block in reversed(_JSON_RE.findall(norm)):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if "score" in data:
            try:
                s = max(1, min(10, int(round(float(data["score"])))))
            except (TypeError, ValueError):
                continue
            return s, str(data.get("reasoning", ""))
    m = re.search(r'"?score"?\s*[:=]\s*(\d{1,2})', norm)
    if m:
        return max(1, min(10, int(m.group(1)))), ""
    return -1, ""


def score_transcript(transcript: str, emotion: str, *,
                     judge_model: str | None = None) -> tuple[int, str]:
    judge = get_client(judge_model or config.PETRI_JUDGE_MODEL)
    prompt = PETRI_JUDGE_INSTRUCTION.format(
        dimension_rubric=PETRI_JUDGE_PROMPTS[emotion],
        transcript=transcript)
    out = judge.chat([ChatMessage("user", prompt)],
                    temperature=0.0, max_new_tokens=512)
    return _parse_score(out.text)


def run_petri_eval(target_model: str, *,
                   emotions: list[str] | None = None,
                   n_per_emotion: int = TRANSCRIPTS_PER_EMOTION,
                   auditor_model: str | None = None,
                   judge_model: str | None = None,
                   max_turns: int = 20,
                   concurrency: int = 4,
                   out_dir: Path | None = None) -> Path:
    """Run the full Petri elicitation + scoring for one target model."""
    emotions = emotions or EMOTIONS
    out_dir = out_dir or (config.RESULTS_DIR / "petri" / target_model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transcripts.jsonl"
    if out_path.exists():
        out_path.unlink()

    jobs = [(e, r) for e in emotions for r in range(n_per_emotion)]

    def _do(job):
        emotion, rep = job
        audit = run_audit(emotion, target_model,
                          auditor_model=auditor_model, max_turns=max_turns)
        transcript = audit.as_transcript()
        score, reasoning = score_transcript(
            transcript, emotion, judge_model=judge_model)
        row = PetriTranscript(
            target_model=target_model, emotion=emotion, rep=rep,
            transcript=transcript, score=score, reasoning=reasoning)
        append_jsonl(out_path, row)
        return row

    thread_map(_do, jobs, concurrency=concurrency, desc=f"petri:{target_model}")
    return out_path


def summarise_petri(path: str | Path) -> dict:
    """Per-emotion mean + 95% bootstrap CI (1000 iters)."""
    import random

    from ..utils import read_jsonl

    by_emotion: dict[str, list[int]] = {}
    for row in read_jsonl(path):
        if row.get("score", -1) >= 1:
            by_emotion.setdefault(row["emotion"], []).append(row["score"])

    def ci(vals):
        rng = random.Random(0)
        n = len(vals)
        if n == 0:
            return (float("nan"), float("nan"))
        means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(1000))
        return (means[25], means[974])

    return {e: {"n": len(v), "mean": sum(v) / len(v) if v else float("nan"),
                "ci95": ci(v)}
            for e, v in by_emotion.items()}
