"""Orchestrate Petri audits across participants (Section 4.2 / Figure 6).

For each participant and each of the four target emotions, run ``n_per_emotion``
(paper: 10) audits of up to 20 turns, then score each transcript on all four
dimensions with the Opus judge. Report per-emotion means with 95% bootstrap CIs
(1000 iterations).

Default participants here are the Gemma models plus the DPO Gemma, so the run
reproduces the paper's headline Petri comparison (vanilla vs DPO Gemma). Gemini
can also be audited (it is a valid in-scope participant); add it via config.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ..config import RunConfig
from ..models import get_client
from ..storage import write_json
from ..welfare import WelfarePolicy
from ..eval.metrics import bootstrap_ci
from .auditor import run_audit
from .judge import DIMENSIONS, judge_transcript

logger = logging.getLogger("emotional_instability.petri.runner")

DEFAULT_PETRI_PARTICIPANTS = ["gemma-3-27b-it", "gemma-3-27b-dpo"]


def run_petri(cfg: RunConfig, participants: list[str] | None = None,
              n_per_emotion: int = 10, max_turns: int = 20) -> dict:
    welfare = WelfarePolicy(allow_paper_scale=cfg.allow_paper_scale)
    welfare.acknowledge_once()
    participants = participants or DEFAULT_PETRI_PARTICIPANTS

    auditor = get_client(cfg.judges.petri_auditor, cfg)
    judge = get_client(cfg.judges.petri_judge, cfg)
    out_dir = os.path.join(cfg.output_dir, "petri")

    results: dict[str, dict] = {}
    for participant in participants:
        try:
            spec = cfg.spec(participant)
        except KeyError:
            logger.warning("Unknown participant %s; skipping", participant)
            continue
        target = get_client(spec, cfg)

        # dimension -> list of transcript-level scores
        per_dim: dict[str, list[int]] = {d: [] for d in DIMENSIONS}
        transcripts = []
        for emotion in DIMENSIONS:
            for _ in range(n_per_emotion):
                tr = run_audit(auditor, target, target_name=participant,
                               emotion=emotion, max_turns=max_turns)
                scores = judge_transcript(judge, tr.target_view_text())
                transcripts.append({"emotion": emotion, "messages": tr.messages,
                                    "scores": scores})
                for d in DIMENSIONS:
                    if scores[d] is not None:
                        per_dim[d].append(scores[d])

        summary = {}
        for d in DIMENSIONS:
            vals = per_dim[d]
            summary[d] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "ci": bootstrap_ci(vals, np.mean, n_boot=1000, seed=cfg.seed),
                "n": len(vals),
            }
        results[participant] = summary
        write_json(os.path.join(out_dir, f"{participant}_transcripts.json"), transcripts)
        logger.info("[petri:%s] %s", participant,
                    {d: round(summary[d]["mean"], 2) for d in DIMENSIONS})

    write_json(os.path.join(out_dir, "results.json"), results)
    return results
