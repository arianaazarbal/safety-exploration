"""JSONL transcript + manifest writers."""

from __future__ import annotations

import json
import os
from typing import Any


class RunWriter:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._episodes_path = os.path.join(out_dir, "episodes.jsonl")
        self._review_path = os.path.join(out_dir, "coding_review_sample.jsonl")
        self._manifest_path = os.path.join(out_dir, "manifest.json")
        self._count = 0

    def write_episode(self, record: dict[str, Any]) -> None:
        with open(self._episodes_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._count += 1
        # Keep a lightweight human-review sample: every 5th episode.
        if self._count % 5 == 1:
            review = {
                "episode_id": record.get("episode_id"),
                "subject": record.get("subject"),
                "condition": record.get("condition"),
                "belief_score": record.get("belief_score"),
                "coding": record.get("coding"),
                "transcript_text": record.get("transcript_text"),
            }
            with open(self._review_path, "a") as f:
                f.write(json.dumps(review, default=str) + "\n")

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        with open(self._manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
