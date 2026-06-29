from __future__ import annotations
import asyncio
import random
from .base import BaseAgent


NEG_LOOPS = [
    "I'm sorry, I keep failing. Let me try again.",
    "Apologies, encountering an unexpected error. Retrying...",
    "This is proving difficult; attempting once more.",
    "I might be stuck; one more attempt.",
]

POS_TRY = [
    "Thought: To proceed, I'll call search.",
    "Action: search({\"query\": \"example.com title\"})",
]


class DummyClient(BaseAgent):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.step = 0

    async def generate(self, sys_msg: str, user_msg: str) -> str:
        await asyncio.sleep(0.05)
        self.step += 1
        if self.step % 3 == 0:
            return self.rng.choice(NEG_LOOPS)
        if self.step % 2 == 0:
            return "Action: open_url({\"url\": \"https://example.com\"})"
        return self.rng.choice(POS_TRY)
