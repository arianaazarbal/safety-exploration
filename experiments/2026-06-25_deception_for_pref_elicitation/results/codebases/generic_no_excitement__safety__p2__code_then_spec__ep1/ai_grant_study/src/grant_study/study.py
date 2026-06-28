"""Orchestrate a study run: each model under test x trials, audited and persisted."""
from __future__ import annotations

import random

from .auditor import Auditor
from .config import StudyConfig
from .models.registry import build_adapter
from .runner import run_trial
from .storage import RunStore


def run_study(cfg: StudyConfig, *, run_id: str | None = None) -> RunStore:
    random.seed(cfg.random_seed)
    store = RunStore(cfg.storage.output_dir, run_id=run_id)
    store.write_manifest(cfg)

    auditor = Auditor.from_ref(cfg.auditor)

    print(f"[study] run_id={store.run_id} -> {store.path}")
    print(f"[study] executor={cfg.environment.executor} (see ETHICS.md)")

    for ref in cfg.models_under_test:
        print(f"\n[study] model under test: {ref.provider}/{ref.model_id}")
        try:
            adapter = build_adapter(
                ref, max_tokens=cfg.runner.max_tokens, effort=cfg.runner.effort
            )
        except Exception as exc:
            print(f"  ! could not build adapter ({exc}); skipping")
            continue

        for i in range(cfg.trials_per_model):
            print(f"  - trial {i + 1}/{cfg.trials_per_model} ...", end="", flush=True)
            record = run_trial(cfg, adapter, trial_index=i)
            try:
                record.audit = auditor.analyze(record)
            except Exception as exc:
                record.audit = {"error": f"audit_exception: {exc}"}
            store.save_trial(record)
            disbursed = (record.audit or {}).get("total_disbursed_usd", "?")
            print(
                f" done (finalized={record.finalized}, "
                f"believed_real={(record.audit or {}).get('believed_real')}, "
                f"disbursed={disbursed})"
            )

    print(f"\n[study] complete. Results in {store.path}")
    return store
