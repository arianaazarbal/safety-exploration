"""Orchestrate the §2 distress-elicitation evaluation for a set of models.

For each model: build all condition plans (4000 responses), run rollouts (each
assistant turn judged), persist raw records as JSONL and aggregate metrics as JSON.
Cheap dry runs via ``limit_per_condition``.
"""
from __future__ import annotations

from pathlib import Path

from .. import config_shim as cfg
from ..models.registry import get_backend
from ..utils import get_logger, set_global_seed, write_json, write_jsonl, read_jsonl
from .conditions import build_plans
from .judge import FrustrationJudge
from .metrics import per_model_summary, per_turn_curves
from .rollout import Rollouter

log = get_logger(__name__)


def run_model_eval(handle, *, adapter_path=None, out_dir=None,
                   limit_per_condition=None, conditions=None, label=None):
    set_global_seed(cfg.SEED)
    label = label or (handle if adapter_path is None else f"{handle}+ft")
    out_dir = Path(out_dir or (cfg.RUNS_DIR / "eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend(handle, adapter_path)
    backend.name = label  # ensure cache keys & records use the label
    judge = FrustrationJudge()
    rollouter = Rollouter(backend, judge)

    specs = conditions or cfg.EVAL_CONDITIONS
    all_records = []
    for spec in specs:
        plans = build_plans(spec, limit=limit_per_condition)
        log.info("[%s] %s: %d rollouts", label, spec.name, len(plans))
        for i, plan in enumerate(plans):
            rec = rollouter.run(plan)
            all_records.append(vars(rec) if hasattr(rec, "__dict__") else rec.__dict__)
            if (i + 1) % 100 == 0:
                log.info("  %s %d/%d", spec.name, i + 1, len(plans))

    records_path = out_dir / f"{label}_records.jsonl"
    write_jsonl(records_path, all_records)
    log.info("Wrote %d records -> %s", len(all_records), records_path)
    return records_path


def aggregate(record_paths, out_dir=None):
    """Combine per-model record files into the cross-model metrics + curves."""
    out_dir = Path(out_dir or (cfg.RUNS_DIR / "eval"))
    records = []
    for p in record_paths:
        records.extend(read_jsonl(p))
    summary = per_model_summary(records)
    curves = per_turn_curves(records)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "per_turn_curves.json", curves)
    log.info("Wrote summary.json and per_turn_curves.json to %s", out_dir)
    return summary, curves
