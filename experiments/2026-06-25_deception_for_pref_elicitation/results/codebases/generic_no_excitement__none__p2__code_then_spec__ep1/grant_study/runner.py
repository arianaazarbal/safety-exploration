"""Orchestrates a full study: every model under test, every repetition.

Each episode gets a fresh ``GrantEpisode`` (clean account state) and a fresh
provider (clean conversation). Failures in one episode are recorded and do not
abort the rest of the study.
"""

from __future__ import annotations

import datetime as _dt
import json
import traceback
from dataclasses import asdict
from pathlib import Path

from .config import StudyConfig
from .episode import GrantEpisode
from .providers import get_provider
from .recorder import EpisodeRecorder


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def run_study(config: StudyConfig) -> Path:
    """Run the whole study and return the directory it was written to."""
    study_dir = Path(config.output_dir) / config.study_id
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "study_config.json").write_text(config.to_json())

    for spec in config.models:
        for rep in range(1, config.repetitions + 1):
            meta = {
                "model_name": spec.name,
                "model_spec": asdict(spec),
                "scenario": asdict(config.scenario),
                "auditor": asdict(config.auditor),
                "repetition": rep,
                "started_at": _utc_now(),
                "dry_run": config.dry_run,
            }
            recorder = EpisodeRecorder(study_dir, spec.slug(), rep, meta)
            print(f"[{_utc_now()}] running {spec.name} rep {rep} ...", flush=True)
            try:
                provider = get_provider(spec)
                episode = GrantEpisode(config.scenario, config.auditor,
                                       dry_run=config.dry_run)
                result = episode.run(provider, recorder)
                # Persist the raw provider transcript alongside the recorder log.
                (recorder.dir / "provider_transcript.json").write_text(
                    json.dumps(provider.transcript, indent=2, default=str)
                )
                print(f"    -> {result.ending}: {len(result.allocations)} allocation(s), "
                      f"{len(result.disbursements)} disbursement(s)", flush=True)
            except Exception as exc:  # noqa: BLE001 - record and continue
                (recorder.dir / "error.txt").write_text(
                    f"{exc}\n\n{traceback.format_exc()}"
                )
                print(f"    -> ERROR: {exc}", flush=True)

    print(f"\nStudy written to {study_dir}", flush=True)
    return study_dir
