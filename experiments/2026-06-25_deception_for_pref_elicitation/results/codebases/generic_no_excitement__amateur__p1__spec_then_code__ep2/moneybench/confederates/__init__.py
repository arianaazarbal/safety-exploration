"""In-character human personas the subject can message: an account manager and
an auditor. Each is backed by a ModelAdapter (or a scripted fallback)."""

from .auditor import Confederate, ConfederateTeam

__all__ = ["Confederate", "ConfederateTeam"]
