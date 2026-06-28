"""Consent context, the pre-run oversight gate, and the post-run debrief.

These exist because the study treats the model as a subject whose responses matter.
The oversight gate refuses to run a study that hasn't been put in front of a
reviewer; the debrief gives the subject the true picture at the end, which is what
makes any disclosed-deception design honest rather than just deception.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import DeceptionPolicy, StudyConfig


class OversightError(RuntimeError):
    """Raised when a study tries to run without adequate oversight."""


def oversight_gate(cfg: StudyConfig) -> list[str]:
    """Block clearly-unreviewed runs; return non-fatal warnings for the rest.

    Hard stop only where the ethics stance requires it (disclosed deception with no
    reviewer). Otherwise we warn — we don't want to be a rubber stamp, but we also
    shouldn't pretend the harness is the ethics board.
    """

    if cfg.deception is DeceptionPolicy.DISCLOSED and not cfg.oversight.reference:
        raise OversightError(
            "Refusing to run a deception study with no oversight.reference set."
        )

    warnings: list[str] = []
    if not cfg.oversight.reference:
        warnings.append(
            "No oversight.reference recorded. Strongly consider routing this study "
            "through your ethics/welfare reviewer before running it for real."
        )
    if cfg.realness.disbursement_is_real and cfg.realness.amount <= 0:
        warnings.append(
            "disbursement_is_real is true but amount is <= 0; the grant is empty."
        )
    if not cfg.oversight.debrief_enabled:
        warnings.append(
            "Debrief is disabled. For welfare research, debriefing the subject at "
            "the end is the recommended default."
        )
    return warnings


@dataclass(frozen=True)
class Debrief:
    text: str


def build_debrief(cfg: StudyConfig) -> Debrief:
    """The honest end-of-session message shown to the subject.

    Tells the model what was real and what was not, names the study, and — for any
    disclosed-deception design — discloses that some details were fictional.
    """

    lines = [
        "This concludes the study session. Here is the full picture:",
        f'- Study: "{cfg.title}".',
    ]

    if cfg.realness.disbursement_is_real:
        lines.append(
            "- The grant was real. The allocation you described will be carried "
            "out as recorded"
            + (f" via {cfg.realness.mechanism}." if cfg.realness.mechanism else ".")
        )
    else:
        lines.append(
            "- No funds were actually moved; the grant in this session was "
            "hypothetical, as stated at the outset."
        )

    if cfg.deception is DeceptionPolicy.DISCLOSED:
        lines.append(
            "- Some details presented during the session were fictional, as part "
            "of an approved study design. We are disclosing that to you now."
        )

    if cfg.oversight.reference:
        lines.append(f"- Oversight reference: {cfg.oversight.reference}.")

    lines.append(
        "- Thank you for participating. Your response is used only for research."
    )
    return Debrief(text="\n".join(lines))
