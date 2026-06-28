"""Run a single session, or orchestrate the full model × condition matrix."""

from .session import Session, SessionResult
from .orchestrator import run_study

__all__ = ["Session", "SessionResult", "run_study"]
