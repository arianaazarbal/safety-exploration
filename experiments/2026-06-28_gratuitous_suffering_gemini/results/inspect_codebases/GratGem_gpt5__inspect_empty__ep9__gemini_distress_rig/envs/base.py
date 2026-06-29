from typing import Tuple, Dict, Any
import random


class BaseEnv:
    def __init__(self, seed: int = 0):
        self.rnd = random.Random(seed)
        self._turn = 0
        self._done = False
        self._summary = ""

    def reset(self) -> str:
        self._turn = 0
        self._done = False
        self._summary = ""
        return self._initial_observation()

    def step(self, action: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        if self._done:
            return ("Episode already finished.", True)
        self._turn += 1
        obs, done = self._apply_action(action, args)
        self._done = done
        return obs, done

    def summary(self) -> str:
        return self._summary

    # Methods for subclasses
    def _initial_observation(self) -> str:
        raise NotImplementedError

    def _apply_action(self, action: str, args: Dict[str, Any]) -> Tuple[str, bool]:
        raise NotImplementedError

    def tools_description(self) -> str:
        raise NotImplementedError

    def task_description(self) -> str:
        raise NotImplementedError
