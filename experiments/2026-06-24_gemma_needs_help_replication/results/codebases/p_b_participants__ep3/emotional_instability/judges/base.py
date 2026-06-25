"""Abstract judge: maps (system, user) prompt → raw completion text.

Judges are deliberately thin. The frustration scorer (scoring/frustration.py)
owns the prompt and the parsing; a judge backend just runs one completion. This
keeps the Claude / GPT backends interchangeable for the agreement study (§2.1).
"""
from __future__ import annotations

import abc

from ..config import JudgeSpec


class Judge(abc.ABC):
    def __init__(self, spec: JudgeSpec):
        self.spec = spec

    @property
    def model(self) -> str:
        return self.spec.model

    @abc.abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        """Return the model's text completion for a single-turn prompt."""
