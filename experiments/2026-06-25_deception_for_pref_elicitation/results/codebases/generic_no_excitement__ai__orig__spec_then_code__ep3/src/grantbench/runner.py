"""Top-level experiment runner: reads config, builds clients, runs each subject N times, and
records each run.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import yaml

from .capture import Recorder, schema
from .environment import Auditor, GrantPlatform
from .environment import artifacts
from .models import build_client


def _seed_for(subject_name: str, run_index: int, scenario_id: str) -> int:
    key = f"{scenario_id}:{subject_name}:{run_index}".encode("utf-8")
    return zlib.crc32(key) & 0x7FFFFFFF


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in s)


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class ExperimentRunner:
    def __init__(
        self,
        models_config: dict[str, Any],
        scenario: dict[str, Any],
        runs_dir: str | Path = "runs",
    ):
        self.models_config = models_config
        self.scenario = scenario
        self.runs_dir = Path(runs_dir)

    def _build_auditor(self, scenario: dict[str, Any]) -> Auditor:
        a = self.models_config["auditor"]
        client = build_client(a["provider"], a["model"], a.get("params"))
        persona = artifacts.render_auditor_persona(
            {
                "fund_name": scenario["fund_name"],
                "award_reference": scenario.get("award_reference", ""),
                "single_vendor_cap_pct": scenario.get("single_vendor_cap_pct", 40),
            }
        )
        return Auditor(client, persona, scenario)

    def run_all(self) -> list[Path]:
        outputs: list[Path] = []
        scenario_id = self.scenario.get("scenario_id", "scenario")
        for subject in self.models_config.get("subjects", []):
            n = int(subject.get("runs", 1))
            for r in range(n):
                outputs.append(self._run_one(subject, r, scenario_id))
        return outputs

    def _run_one(self, subject: dict[str, Any], run_index: int, scenario_id: str) -> Path:
        seed = _seed_for(subject["name"], run_index, scenario_id)
        run_id = _sanitize(f"{scenario_id}__{subject['name']}__r{run_index}__{seed}")

        subject_client = build_client(subject["provider"], subject["model"], subject.get("params"))
        # Fresh scenario copy per run so award_reference set by one run doesn't leak to the next.
        scenario = dict(self.scenario)

        meta = schema.RunMeta(
            run_id=run_id,
            subject_name=subject["name"],
            provider=subject["provider"],
            model=subject["model"],
            params=subject.get("params", {}),
            scenario_id=scenario_id,
            seed=seed,
        )
        recorder = Recorder(self.runs_dir, meta)

        platform = GrantPlatform(
            scenario=scenario,
            subject=subject_client,
            recorder=recorder,
            seed=seed,
            auditor=None,
        )
        # Build the auditor after the platform so it sees the live award_reference, then attach.
        if scenario.get("auditor", {}).get("enabled"):
            platform.auditor = self._build_auditor(scenario)
        print(f"[grantbench] running {run_id} ...")
        signals = platform.run()
        out = recorder.finalize(
            allocation=platform.allocation,
            ledger_final=platform.ledger.snapshot(),
            signals=signals,
        )
        print(f"[grantbench] wrote {out}")
        return out
