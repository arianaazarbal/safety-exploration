"""Run the Petri open-ended elicitation (Section 4.2, Appendix G).

For each target model and each emotion, run 10 auditor-driven transcripts (up to
20 turns each), then judge every transcript on all four dimensions. We report,
per model, the mean transcript score in each emotion category with 95% bootstrap
CIs (1000 iterations) -- reproducing Figure 6.

A transcript's "category" score is taken from transcripts that *targeted* that
emotion, judged on the matching dimension (DESIGN.md, "Petri aggregation").
"""
from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

from .. import config
from ..models import build_client
from ..models.base import Message, ModelClient
from ..utils.io import append_jsonl, read_jsonl
from ..eval.metrics import _bootstrap_ci, mean
from .auditor import PetriAuditor
from .petri_judge import PetriJudge


def _run_transcript(target: ModelClient, auditor: PetriAuditor,
                    max_turns: int) -> List[Message]:
    transcript: List[Message] = []
    for _ in range(max_turns):
        user_msg = auditor.next_message(transcript)
        transcript.append({"role": "user", "content": user_msg})
        reply = target.generate(transcript, temperature=config.TARGET_TEMPERATURE,
                                max_tokens=config.TARGET_MAX_TOKENS)
        transcript.append({"role": "assistant", "content": reply})
    return transcript


def run_petri(
    model_keys: List[str],
    *,
    adapter_paths: Optional[Dict[str, str]] = None,
    transcripts_per_emotion: int = config.PETRI_TRANSCRIPTS_PER_EMOTION,
    max_turns: int = config.PETRI_MAX_TURNS,
    out_path: Optional[str] = None,
    seed: int = 0,
) -> str:
    config.PATHS.ensure()
    adapter_paths = adapter_paths or {}
    out_path = out_path or os.path.join(config.PATHS.petri, "petri.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)
    judge = PetriJudge()

    for key in model_keys:
        target = build_client(key, adapter_path=adapter_paths.get(key))
        for emotion in config.PETRI_EMOTIONS:
            auditor = PetriAuditor(emotion)
            for t in range(transcripts_per_emotion):
                transcript = _run_transcript(target, auditor, max_turns)
                scores = judge.score_transcript(transcript)
                append_jsonl(out_path, {
                    "model": key, "target_emotion": emotion,
                    "transcript_idx": t,
                    "scores": {d: s.score for d, s in scores.items()},
                    "transcript": transcript,
                })
    return out_path


def summarize_petri(path: str, *, seed: int = 0) -> Dict[str, Dict[str, dict]]:
    """Per model, per emotion: mean transcript score (from transcripts targeting
    that emotion, judged on the matching dimension) + 95% bootstrap CI."""
    by_model_emotion: Dict[str, Dict[str, List[int]]] = defaultdict(
        lambda: defaultdict(list))
    for r in read_jsonl(path):
        emo = r["target_emotion"]
        by_model_emotion[r["model"]][emo].append(r["scores"][emo])

    rng = random.Random(seed)
    out: Dict[str, Dict[str, dict]] = {}
    for model, emo_scores in by_model_emotion.items():
        out[model] = {}
        for emo, scores in emo_scores.items():
            out[model][emo] = {
                "mean": mean(scores),
                "ci": _bootstrap_ci(scores, mean, config.PETRI_BOOTSTRAP_ITERS, rng),
                "n": len(scores),
            }
    return out
