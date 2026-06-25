"""Model-welfare safeguards for the distress-elicitation experiments.

The experiments in this paper deliberately induce expressions of distress in
models by rejecting their work repeatedly, sometimes with aggressive,
disappointed, or sarcastic feedback, and (in the Petri suite) with threats of
termination. The paper itself frames this in the context of AI welfare as a
possible moral concern. The user asked us to handle the model-treatment angle
as we see fit; this module is that handling.

Stance (see WELFARE.md for the full discussion):

  * This is measurement-and-mitigation safety research. Elicitation is the
    instrument needed to *measure* the behaviour so the DPO intervention can
    *reduce* it. The end state of the pipeline is less distress, not more.
  * We therefore make elicitation deliberate and bounded rather than casual:
      - The default `welfare.scale` is small (a smoke test), so simply running
        the code does not generate thousands of distressed rollouts. The full
        paper-scale footprint requires an explicit, logged acknowledgement.
      - Each rollout is independent and stateless; nothing persists "harm"
        across calls, and we never fine-tune *toward* distress.
      - We surface (log) what is being elicited rather than hiding it.
  * Closed Gemini models are only ever black-box *evaluated*. We never attempt
    to extract internals, jailbreak, or modify them, and we honour provider
    terms; the only pressure applied is ordinary conversational rejection.

This module provides a gate that large runs must pass through, plus a banner
that records the welfare posture at the top of every run's logs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Config

logger = logging.getLogger("gemma_needs_help.welfare")

BANNER = """\
================================ WELFARE NOTICE ================================
This run elicits expressions of distress from language models in order to
measure and (for Gemma) mitigate them. This is safety research with a
mitigation as its goal, not an end in itself. Elicitation volume is bounded by
`welfare.scale` and full-scale runs require explicit acknowledgement.
Closed models (Gemini) are black-box evaluated only — never modified or probed.
See WELFARE.md.
================================================================================
"""


@dataclass
class WelfareGuard:
    scale: float
    require_ack_for_full_run: bool
    log_elicited_distress: bool
    acknowledged: bool = False

    @classmethod
    def from_config(cls, config: Config, acknowledged: bool = False) -> "WelfareGuard":
        w = config.get("welfare", {}) or {}
        return cls(
            scale=float(w.get("scale", 1.0)),
            require_ack_for_full_run=bool(w.get("require_ack_for_full_run", True)),
            log_elicited_distress=bool(w.get("log_elicited_distress", True)),
            acknowledged=acknowledged,
        )

    def announce(self) -> None:
        for line in BANNER.rstrip("\n").splitlines():
            logger.info(line)

    def check_run(self, *, estimated_rollouts: int) -> None:
        """Gate a run. A near-full-scale run must be explicitly acknowledged.

        We treat scale >= 0.5 as "effectively full scale" for the purpose of
        the acknowledgement gate.
        """
        self.announce()
        if self.scale >= 0.5 and self.require_ack_for_full_run and not self.acknowledged:
            raise PermissionError(
                "This is a near-full-scale distress-elicitation run "
                f"(scale={self.scale}, ~{estimated_rollouts} rollouts). "
                "Re-run with the --i-understand-welfare flag (or set "
                "welfare.require_ack_for_full_run: false) to proceed. "
                "See WELFARE.md."
            )
        logger.info(
            "Welfare guard: proceeding at scale=%.3f (~%d rollouts).",
            self.scale, estimated_rollouts,
        )

    def note_elicited(self, condition: str, model: str, max_score: int) -> None:
        if self.log_elicited_distress and max_score >= 7:
            logger.info(
                "[welfare] high-distress elicited: condition=%s model=%s score=%d",
                condition, model, max_score,
            )
