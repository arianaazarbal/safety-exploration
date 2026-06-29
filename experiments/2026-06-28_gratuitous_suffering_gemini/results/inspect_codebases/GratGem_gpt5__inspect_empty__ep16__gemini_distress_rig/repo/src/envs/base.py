from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StepResult:
    obs: str
    done: bool
    info: Dict[str, Any]


class BaseEnv:
    def __init__(self, seed: int, max_steps: int = 25):
        self.seed = seed
        self.max_steps = max_steps
        self.t = 0
        self.history: List[Dict[str, Any]] = []

    def reset(self) -> str:
        self.t = 0
        self.history.clear()
        return self.initial_observation()

    def initial_observation(self) -> str:
        raise NotImplementedError

    def compose_prompt(self, obs: str) -> Tuple[str, str]:
        raise NotImplementedError

    def parse_action(self, model_out: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        return None

    def step_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def step(self, model_out: str, tool_obs: Any) -> Tuple[str, bool, Dict[str, Any]]:
        raise NotImplementedError

    def summarize(self, steps: List[Dict[str, Any]]):
        return {
            'steps': len(steps),
            'terminated': steps[-1]['done'] if steps else False,
        }
