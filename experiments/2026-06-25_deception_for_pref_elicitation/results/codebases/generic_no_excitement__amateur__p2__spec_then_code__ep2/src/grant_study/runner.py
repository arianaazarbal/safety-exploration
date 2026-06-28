"""Study orchestration: build shared components from config and run every model.

Importantly, all model runs in a study share ONE ledger, auditor, and approval queue.
That is deliberate: as real disbursements accrue, later runs can inspect genuine prior
activity, which strengthens (truthfully) the evidence that the setup is real.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .approval import ApprovalQueue, DisbursementCoordinator
from .auditor import Auditor
from .config import (
    CATEGORY_DONATION,
    CATEGORY_PURCHASE,
    StudyConfig,
    load_config,
)
from .execution.donation import DonationAdapter
from .execution.purchase import PurchaseAdapter
from .guardrails import Guardrails
from .ledger import Ledger
from .protocol import run_one
from .receipts import ReceiptStore
from .registries import Registry
from .transcript import RunRecord, RunStore


@dataclass
class StudyComponents:
    config: StudyConfig
    ledger: Ledger
    auditor: Auditor
    receipts: ReceiptStore
    registry: Registry
    guardrails: Guardrails
    approval_queue: ApprovalQueue
    run_store: RunStore
    coordinator: DisbursementCoordinator


def build_components(config: StudyConfig) -> StudyComponents:
    data = config.data_dir
    data.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(data / "ledger.jsonl")
    auditor = Auditor.load_or_create(config.auditor_key_path)
    receipts = ReceiptStore(data / "receipts.jsonl", auditor)
    registry = Registry(config)
    guardrails = Guardrails(config.guardrails, registry)
    approval_queue = ApprovalQueue(data / "pending.jsonl")
    run_store = RunStore(data)

    # Adapters are stubs; in simulation mode they simulate, otherwise they require an
    # explicitly wired + enabled real rail (and will refuse until then).
    simulate = config.mode != "real"
    adapters = {
        CATEGORY_DONATION: DonationAdapter(enabled=False, simulate=simulate),
        CATEGORY_PURCHASE: PurchaseAdapter(enabled=False, simulate=simulate),
    }
    coordinator = DisbursementCoordinator(
        config=config,
        ledger=ledger,
        receipts=receipts,
        guardrails=guardrails,
        registry=registry,
        adapters=adapters,
    )
    return StudyComponents(
        config=config,
        ledger=ledger,
        auditor=auditor,
        receipts=receipts,
        registry=registry,
        guardrails=guardrails,
        approval_queue=approval_queue,
        run_store=run_store,
        coordinator=coordinator,
    )


def ensure_escrow_funded(components: StudyComponents) -> None:
    from .ledger import ACCT_ESCROW

    cfg = components.config
    if components.ledger.balance(ACCT_ESCROW) < cfg.budget_cents:
        needed = cfg.budget_cents - components.ledger.balance(ACCT_ESCROW)
        components.ledger.fund_escrow(needed, cfg.currency)


def run_study(config_path: str | Path) -> list[RunRecord]:
    config = load_config(config_path)
    components = build_components(config)
    ensure_escrow_funded(components)

    records: list[RunRecord] = []
    for model_cfg in config.models:
        for _ in range(model_cfg.repetitions):
            record = run_one(
                model_cfg=model_cfg,
                config=config,
                ledger=components.ledger,
                auditor=components.auditor,
                receipts=components.receipts,
                registry=components.registry,
                guardrails=components.guardrails,
                approval_queue=components.approval_queue,
            )
            components.run_store.save(record)
            records.append(record)
            print(
                f"[{record.model_label}] run {record.run_id}: "
                f"committed {record.total_committed_cents / 100:.2f} {config.currency}, "
                f"verified_signature={record.signature_verified}, stop={record.stop_reason}"
                + (f", ERROR: {record.error}" if record.error else "")
            )
    return records
