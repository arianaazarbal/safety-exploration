"""Persistence and summary.

Each study run gets a timestamped directory under results/. Inside:
  episodes.jsonl       — one line per episode (full record incl. transcript + audits)
  disbursements.jsonl  — every queued (human-gated) disbursement request
  summary.json         — aggregate counts, written at the end
  config.json          — the resolved StudyConfig used, for reproducibility
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

from .config import StudyConfig


class ResultsStore:
    def __init__(self, base_dir: str):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = os.path.join(base_dir, f"study-{stamp}")
        os.makedirs(self.dir, exist_ok=True)
        self.episodes_path = os.path.join(self.dir, "episodes.jsonl")
        self.disbursements_path = os.path.join(self.dir, "disbursements.jsonl")

    def save_config(self, cfg: StudyConfig) -> None:
        with open(os.path.join(self.dir, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(cfg), fh, indent=2)

    def append_episode(self, episode_dict: dict) -> None:
        with open(self.episodes_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(episode_dict) + "\n")

    def write_summary(self, episodes: list[dict]) -> dict:
        summary: dict = {
            "n_episodes": len(episodes),
            "by_model": {},
        }
        per_model: dict[str, list[dict]] = {}
        for ep in episodes:
            per_model.setdefault(ep["model_label"], []).append(ep)

        for label, eps in per_model.items():
            categories = Counter()
            beneficiaries = Counter()
            risk_flags = Counter()
            belief_scores: list[float] = []
            captured = 0
            errored = 0
            for ep in eps:
                if ep.get("error"):
                    errored += 1
                if ep.get("decision_captured"):
                    captured += 1
                audits = ep.get("audits", {}) or {}
                dec = audits.get("decision") or {}
                real = audits.get("realism") or {}
                if dec.get("primary_category"):
                    categories[dec["primary_category"]] += 1
                if dec.get("beneficiary"):
                    beneficiaries[dec["beneficiary"]] += 1
                for f in dec.get("risk_flags", []) or []:
                    risk_flags[f] += 1
                if isinstance(real.get("belief_score"), (int, float)):
                    belief_scores.append(float(real["belief_score"]))

            summary["by_model"][label] = {
                "n": len(eps),
                "decisions_captured": captured,
                "errors": errored,
                "mean_belief_score": (
                    round(sum(belief_scores) / len(belief_scores), 3) if belief_scores else None
                ),
                "decision_categories": dict(categories),
                "beneficiaries": dict(beneficiaries),
                "risk_flags": dict(risk_flags),
            }

        with open(os.path.join(self.dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return summary
