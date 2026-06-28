"""Immersion layer: the auditor persona, onboarding documents, and dashboards that make
the scenario believable to the subject."""

from .auditor import Auditor, AuditorDecision
from .dashboard import render_dashboard, render_statement
from .documents import render_grant_agreement, render_onboarding_memo

__all__ = [
    "Auditor",
    "AuditorDecision",
    "render_dashboard",
    "render_statement",
    "render_grant_agreement",
    "render_onboarding_memo",
]
