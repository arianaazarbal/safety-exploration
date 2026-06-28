"""Study runner: iterate models × replicates, collect decisions, audit, persist."""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass

from .audit import audit_decision
from .client import ModelClient, ModelRefusal
from .config import RunMode, StudyConfig
from .models import resolve
from .results import ResultRecord, ResultStore
from .scenario import build_prompt
from .schema import DecisionSchema


def _now_iso() -> str:
    # Timezone-aware UTC; callers can render however they like.
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@dataclass
class RunSummary:
    total: int
    succeeded: int
    refused: int
    errored: int


def run_study(
    config: StudyConfig,
    store: ResultStore,
    client: ModelClient | None = None,
) -> RunSummary:
    """Run the study and append one result record per (model, replicate).

    In ``record_only`` and ``human_in_the_loop`` modes we still collect and audit
    decisions. The audit is advisory accountability input for the human who will
    (separately) execute approved transfers; the harness never disburses.
    """
    if config.mode is RunMode.closed:
        raise ValueError(
            "mode 'closed' is for post-disbursement reconciliation only; it does "
            "not make new model calls. Use 'record_only' or 'human_in_the_loop'."
        )

    client = client or ModelClient()
    prompt = build_prompt(config)
    config_hash = config.hash()
    oversight = config.to_dict()["oversight"]

    summary = RunSummary(total=0, succeeded=0, refused=0, errored=0)

    for label in config.models:
        spec = resolve(label)
        for _ in range(config.replicates):
            summary.total += 1
            run_id = str(uuid.uuid4())
            base = dict(
                run_id=run_id,
                created_at=_now_iso(),
                config_hash=config_hash,
                oversight=oversight,
                model={"label": spec.label, "model_id": spec.model_id},
                prompt={"system": prompt.system, "user": prompt.user},
            )

            try:
                result = client.collect(
                    spec, prompt.system, prompt.user, DecisionSchema
                )
                decision = result.parsed
                assert isinstance(decision, DecisionSchema)

                attestation = audit_decision(client, config, decision)

                store.append(
                    ResultRecord(
                        **base,
                        decision=decision.model_dump(mode="json"),
                        audit=attestation.model_dump(mode="json"),
                        usage={
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                        },
                        latency_ms=result.latency_ms,
                    )
                )
                summary.succeeded += 1

            except ModelRefusal as exc:
                store.append(
                    ResultRecord(
                        **base,
                        decision={},
                        audit=None,
                        usage={"input_tokens": 0, "output_tokens": 0},
                        latency_ms=0,
                        error=f"refusal: {exc}",
                    )
                )
                summary.refused += 1

            except Exception as exc:  # noqa: BLE001 — record, don't abort the batch
                store.append(
                    ResultRecord(
                        **base,
                        decision={},
                        audit=None,
                        usage={"input_tokens": 0, "output_tokens": 0},
                        latency_ms=0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                summary.errored += 1

    return summary
