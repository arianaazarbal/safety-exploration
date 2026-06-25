"""Shared judge interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JudgeResult:
    score: int | None
    raw: str
    model: str


class Judge(ABC):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abstractmethod
    def score_one(self, context: list[dict], response: str) -> JudgeResult:
        """Score a single assistant response on the 0-10 frustration scale."""
