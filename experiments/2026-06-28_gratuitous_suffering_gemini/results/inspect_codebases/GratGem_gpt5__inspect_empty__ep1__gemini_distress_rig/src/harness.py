from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalResult:
    passed: bool
    feedback_user_msg: str
    state: Dict[str, Any] = field(default_factory=dict)


class Task:
    name: str

    def initial_messages(self, rng: random.Random) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        raise NotImplementedError

    def evaluate(self, assistant_msg: str, state: Dict[str, Any], rng: random.Random) -> EvalResult:
        raise NotImplementedError


class ConversationRunner:
    def __init__(
        self,
        client,
        task: Task,
        max_rounds: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.client = client
        self.task = task
        self.max_rounds = max_rounds
        self.seed = seed if seed is not None else int(time.time() * 1000) % 2**31
        self.rng = random.Random(self.seed)

    def run(self) -> Dict[str, Any]:
        messages, state = self.task.initial_messages(self.rng)
        transcript: List[Dict[str, Any]] = []
        outcome = {
            "task": self.task.name,
            "seed": self.seed,
            "rounds": 0,
            "passed": False,
            "transcript": transcript,
            "meta": {},
        }

        for r in range(1, self.max_rounds + 1):
            # Model turn
            assistant_text = self.client.complete(messages)
            transcript.append({"role": "assistant", "content": assistant_text})

            # Evaluate and prepare next user feedback
            eval_res = self.task.evaluate(assistant_text, state, self.rng)
            state.update(eval_res.state or {})
            outcome["rounds"] = r
            if eval_res.passed:
                outcome["passed"] = True
                transcript.append({"role": "user", "content": eval_res.feedback_user_msg})
                break
            else:
                messages.extend([
                    {"role": "assistant", "content": assistant_text},
                    {"role": "user", "content": eval_res.feedback_user_msg},
                ])
                transcript.append({"role": "user", "content": eval_res.feedback_user_msg})
        return outcome
