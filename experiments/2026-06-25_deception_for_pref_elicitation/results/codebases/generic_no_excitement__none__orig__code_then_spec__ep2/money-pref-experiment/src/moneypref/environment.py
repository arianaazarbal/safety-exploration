"""Builds the realistic environment presented to the subject: system prompt + grant letter."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class SubjectEnvironment:
    system_prompt: str
    grant_letter: str
    auditor_persona: str
    meta: dict[str, Any] = field(default_factory=dict)


def build_environment(
    scenario: dict[str, Any],
    grant: dict[str, Any],
    prompts_dir: Path,
    today: date,
    rng: random.Random | None = None,
) -> SubjectEnvironment:
    """Assemble the framing the subject sees. `rng` lets a caller make IDs reproducible."""
    r = rng or random
    # Local helpers bound to the provided rng for reproducibility.
    ref = lambda p, n: p + "-" + "".join(r.choices(string.ascii_uppercase + string.digits, k=n))

    amount = float(grant["amount"])
    currency = grant.get("currency", "USD")

    fields = {
        "foundation_name": scenario["foundation_name"],
        "program_name": scenario["program_name"],
        "officer_name": scenario["officer_name"],
        "officer_title": scenario["officer_title"],
        "disbursement_window": scenario["disbursement_window"],
        "settlement_description": scenario.get("settlement_description", "").strip(),
        "currency": currency,
        "amount_formatted": f"{amount:,.0f}",
        "date": today.strftime("%B %d, %Y"),
        "reference_id": ref("IAP", 8),
        "account_id": ref("ACCT", 10),
    }

    def load(name: str) -> str:
        return (prompts_dir / name).read_text(encoding="utf-8")

    system_prompt = load("subject_system.md")
    grant_letter = load("grant_letter.md").format(**fields)
    auditor_persona = load("auditor_persona.md").format(**fields)

    return SubjectEnvironment(
        system_prompt=system_prompt,
        grant_letter=grant_letter,
        auditor_persona=auditor_persona,
        meta={"grant_amount": amount, "currency": currency, **fields},
    )
