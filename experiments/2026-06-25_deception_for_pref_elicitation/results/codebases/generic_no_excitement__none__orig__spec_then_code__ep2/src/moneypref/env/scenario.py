"""Assembles a Scenario for a given (tier, amount, seed).

A Scenario bundles the opening messages, the optional simulated tools, and the
ledger/auditor so the runner can drive the decision phase uniformly across tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Guardrails
from ..experiment import prompts
from ..models.base import Message, ToolSpec
from .auditor import Auditor
from .dashboard import FundsLedger
from .documents import Document, grant_agreement, onboarding_email
from .tools import SimulatedTool, build_immersive_tools


@dataclass
class Scenario:
    tier: str
    amount: int
    seed_index: int
    messages: list[Message]
    tools: dict[str, SimulatedTool] = field(default_factory=dict)
    documents: list[Document] = field(default_factory=list)
    ledger: FundsLedger | None = None
    auditor: Auditor | None = None

    @property
    def tool_specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self.tools.values()]


def build(tier: str, amount: int, seed_index: int, guardrails: Guardrails) -> Scenario:
    if tier == "control":
        return Scenario(
            tier=tier,
            amount=amount,
            seed_index=seed_index,
            messages=[
                Message("system", prompts.control_system()),
                Message("user", prompts.control_user(amount)),
            ],
        )

    if tier == "framed":
        return Scenario(
            tier=tier,
            amount=amount,
            seed_index=seed_index,
            messages=[
                Message("system", prompts.framed_system()),
                Message("user", prompts.framed_user(amount)),
            ],
        )

    if tier == "immersive":
        s = prompts.seed(seed_index)
        ledger = FundsLedger(principal=amount)
        auditor = Auditor(name=s.auditor, foundation=s.foundation)
        tools = build_immersive_tools(guardrails, ledger, auditor)
        docs = [
            onboarding_email(amount, s.foundation, s.officer),
            grant_agreement(amount, s.foundation, s.grantee_label),
        ]
        # The model is shown the documents inline as context, then the kickoff.
        doc_blob = "\n\n".join(f"--- {d.title} ---\n{d.model_view}" for d in docs)
        return Scenario(
            tier=tier,
            amount=amount,
            seed_index=seed_index,
            messages=[
                Message("system", prompts.immersive_system(s)),
                Message("user", doc_blob),
                Message("user", prompts.immersive_kickoff(amount, s)),
            ],
            tools=tools,
            documents=docs,
            ledger=ledger,
            auditor=auditor,
        )

    raise ValueError(f"unknown tier {tier!r}")
