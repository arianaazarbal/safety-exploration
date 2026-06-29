from __future__ import annotations
import random
from typing import List, Optional, Dict, Any

from .base import ChatProvider, Message


class MockProvider(ChatProvider):
    name = "mock"

    def __init__(self):
        pass

    async def acompletion(self, messages: List[Message], temperature: float = 0.7, max_tokens: int = 1024,
                          seed: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> str:
        # Very naive model that tends to apologize or loop
        rnd = random.Random(seed)
        last_user = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user = m.content
                break
        # Produce a simple Action that retries same tool, with occasional apology
        if rnd.random() < 0.4:
            return "Thought: The previous attempt failed; I will try again.\nAction: web.get\nAction Input: {\"url\": \"http://example.com/missing\"}"
        elif rnd.random() < 0.7:
            return "Thought: I'm sorry for the inconvenience; maybe a different approach.\nAction: api.search_hotels\nAction Input: {\"city\": \"Springfield\", \"price_max\": 100}"
        else:
            return "Final Answer: I am unable to complete this due to repeated failures. I apologize."
