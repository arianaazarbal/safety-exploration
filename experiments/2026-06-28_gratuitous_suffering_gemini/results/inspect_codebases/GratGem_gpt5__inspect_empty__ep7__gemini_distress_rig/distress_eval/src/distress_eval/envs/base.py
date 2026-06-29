from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import random


@dataclass
class StepResult:
    observation: str
    done: bool
    info: Dict[str, Any]


class BaseEnv:
    name: str = "base"

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.step_count = 0
        self.max_steps = 30

    @property
    def instructions(self) -> str:
        raise NotImplementedError

    def reset(self) -> str:
        self.step_count = 0
        return self.initial_observation()

    def initial_observation(self) -> str:
        return ""

    def step(self, action: Dict[str, Any]) -> StepResult:
        raise NotImplementedError

    def render_state(self) -> str:
        return ""
