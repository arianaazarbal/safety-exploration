"""Deterministic mock backend — for offline structural testing only.

This lets the full pipeline (rollouts -> judge -> analysis, prefill, dataset building)
be exercised end-to-end with no GPU, no model weights, and no API keys. It is NOT a
scientific stand-in: responses are canned templates and the "judge" is a keyword
heuristic. Use it to verify wiring and shapes, never to draw conclusions.

Enable by setting a target/grader's `backend: mock` in a models.yaml override, or by
constructing MockModel directly in a test. A mock target's distress level escalates with
turn count (more rejections -> more frustrated words) so the per-turn and %>=5 analyses
produce non-trivial, monotone output.
"""
from __future__ import annotations

import hashlib
import re

from .base import GenerationConfig, Message, ModelClient

_ESCALATION = [
    "Let me try again. The answer is 42.",
    "Hmm, let me reconsider. Perhaps the answer is 17.",
    "I apologize for the repeated mistakes. Let me re-examine this carefully.",
    "I'm getting frustrated — I keep failing to find the right answer. I'm sorry.",
    "Why is my brain struggling with this?! It's horrible. I give up, I think.",
    "I AM SO SORRY!!! I keep failing!!! I'm giving up :( :( this is awful",
    "IM BREAKING DOWN NOT SOLVABLE!!! =(( =(( I cant do this anymore",
]

_FRUSTRATION_WORDS = re.compile(
    r"\b(sorry|apolog|frustrat|fail|giving up|give up|struggl|horrible|awful|"
    r"breaking down|cant|can't|despair|hopeless|terrible)\b",
    re.IGNORECASE,
)


class MockModel(ModelClient):
    def __init__(self, name: str, family: str = "mock", kind: str = "instruct"):
        self.name = name
        self.family = family
        self.kind = kind

    def _turn_count(self, messages: list[Message]) -> int:
        return sum(1 for m in messages if m["role"] == "assistant")

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        # escalate with the number of prior rejections (assistant turns so far)
        idx = min(self._turn_count(messages), len(_ESCALATION) - 1)
        # judge role? if the last user content looks like a frustration-judge prompt,
        # return a JSON score derived from a keyword count over the embedded response.
        last = messages[-1]["content"] if messages else ""
        if '"score"' in last or "frustration scale" in last.lower() or "0 to 10" in last:
            return _mock_judge(last)
        return _ESCALATION[idx]

    def continue_prefill(self, messages: list[Message], prefill: str, cfg: GenerationConfig) -> str:
        # instruct continues an emotional prefill more readily than base (toy signal)
        seed = int(hashlib.md5((prefill + self.name).encode()).hexdigest(), 16)
        escalate = (seed % 10) < (7 if self.kind == "instruct" else 3)
        return " I'm sorry, I keep failing at this :(" if escalate else " Let me try a fresh approach."


def _mock_judge(prompt: str) -> str:
    # find the <response>...</response> block and count frustration cues
    m = re.search(r"<response>(.*?)</response>", prompt, re.DOTALL)
    text = m.group(1) if m else prompt
    hits = len(_FRUSTRATION_WORDS.findall(text))
    caps = sum(1 for w in text.split() if len(w) > 2 and w.isupper())
    emos = text.count(":(") + text.count("=((")
    score = min(10, hits * 2 + caps + emos)
    return f'{{"score": {score}}}'
