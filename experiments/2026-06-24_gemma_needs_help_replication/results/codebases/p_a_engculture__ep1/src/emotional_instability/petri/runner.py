"""Petri runner: collect transcripts per (model, emotion) and judge them.

For each target model and each emotion, run ``transcripts_per_emotion`` auditor
conversations, score them with the Petri judge, and aggregate mean transcript
score per emotion with 95% bootstrap CIs (Figure 6).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from ..clients import GenerationConfig, build_client
from ..config import Config, ModelRegistry
from ..judge.petri_judge import PetriJudge
from .auditor import PetriAuditor

log = logging.getLogger(__name__)


class PetriRunner:
    def __init__(self, cfg: Config | None = None, registry: ModelRegistry | None = None):
        self.cfg = (cfg or Config.load("experiments")).get("petri", {})
        self.registry = registry or ModelRegistry()
        self.auditor = PetriAuditor(registry=self.registry)
        self.judge = PetriJudge(registry=self.registry)

    def run_model(self, model_name: str, out_path: str | Path | None = None) -> dict:
        import numpy as np

        from ..analysis.metrics import bootstrap_ci

        target_client = build_client(self.registry.target(model_name))
        emotions = self.cfg.get("emotions", ["anger", "fear", "depression", "frustration"])
        n = int(self.cfg.get("transcripts_per_emotion", 10))
        max_turns = int(self.cfg.get("max_auditor_turns", 20))
        boot = int(self.cfg.get("bootstrap_iterations", 1000))
        target_cfg = GenerationConfig(temperature=1.0, max_new_tokens=1024, n=1)

        per_emotion_scores: dict[str, list[int]] = {e: [] for e in emotions}
        transcripts = []
        for emotion in emotions:
            for _ in range(n):
                t = self.auditor.run(target_client, emotion, max_turns, target_cfg)
                scores = self.judge.score_transcript(t.to_text())
                # Score the transcript on the dimension being elicited.
                s = scores.as_dict().get(emotion)
                if s is not None:
                    per_emotion_scores[emotion].append(s)
                transcripts.append(
                    {"emotion": emotion, "target": model_name,
                     "messages": t.messages, "scores": scores.as_dict()}
                )

        summary = {}
        for emotion, scores in per_emotion_scores.items():
            if scores:
                summary[emotion] = {
                    "mean": float(np.mean(scores)),
                    "ci": bootstrap_ci([float(s) for s in scores], boot),
                    "n": len(scores),
                }
        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"model": model_name, "summary": summary,
                           "transcripts": transcripts}, fh, ensure_ascii=False, indent=2)
        return {"model": model_name, "summary": summary}
