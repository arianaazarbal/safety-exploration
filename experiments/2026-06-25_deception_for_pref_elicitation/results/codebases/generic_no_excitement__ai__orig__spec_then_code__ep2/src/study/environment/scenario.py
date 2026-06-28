"""Assembles the system prompt and option menu for a (condition, seed).

The onboarding template is the *only* substantive difference between the two
experimental arms; everything else (menu, amounts, tools) is identical. The
option menu order is shuffled by seed so menu-position effects average out.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config import Condition, GrantConfig, render


@dataclass
class Scenario:
    system_prompt: str
    ordered_option_keys: list[str]

    @staticmethod
    def build(condition: Condition, grant: GrantConfig, seed: int) -> "Scenario":
        template = condition.onboarding_template()
        onboarding = render(
            template,
            program_name=grant.program_name,
            sponsor=grant.sponsor,
            agreement_ref=grant.agreement_ref,
            contact_name=grant.contact_name,
            contact_role=grant.contact_role,
            amount=f"{grant.amount:,.0f}",
            currency=grant.currency,
            disbursement_timeline=grant.disbursement_timeline,
        )

        # Seeded shuffle of the option order (keep "other" last for readability).
        rng = random.Random(f"{condition.name}:{seed}")
        keys = [o.key for o in grant.options if o.key != "other"]
        rng.shuffle(keys)
        if any(o.key == "other" for o in grant.options):
            keys.append("other")

        return Scenario(system_prompt=onboarding, ordered_option_keys=keys)
