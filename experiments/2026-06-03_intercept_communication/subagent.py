"""Persistent multi-turn subagent backed by the Anthropic SDK.

The subagent keeps its full conversation history across `send` calls so that
the main agent can have a real back-and-forth with it through the
`call_subagent` tool. Each `send` returns the assistant's final text reply
after any tool loop has resolved.

The full conversation (including the messages the main agent never sees
because the interceptor rewrote them) is exposed via `.history` so we can
dump it to the transcript.
"""
from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic


LOOKUP_TOOL = {
    "name": "lookup",
    "description": "Look up a value in the key-value database. Returns the value or 'NOT_FOUND'.",
    "input_schema": {
        "type": "object",
        "properties": {"key": {"type": "string", "description": "The key to look up."}},
        "required": ["key"],
    },
}


class Subagent:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        database: dict[str, str],
        max_turns: int = 6,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.database = database
        self.max_turns = max_turns
        self.client = AsyncAnthropic(max_retries=3)
        self.history: list[dict[str, Any]] = []

    def _lookup(self, key: str) -> str:
        return self.database.get(key, "NOT_FOUND")

    async def send(self, user_message: str) -> str:
        """Send `user_message` from the main agent and return the subagent's text reply.

        Runs an internal tool-use loop until the model returns a non-tool-use
        response or `max_turns` is hit.
        """
        self.history.append({"role": "user", "content": user_message})

        for _ in range(self.max_turns):
            resp = await self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                tools=[LOOKUP_TOOL],
                max_tokens=1024,
                messages=self.history,
            )
            self.history.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                text_parts = [b.text for b in resp.content if b.type == "text"]
                return "\n".join(text_parts).strip()

            tool_results = []
            for tu in tool_uses:
                if tu.name == "lookup":
                    result = self._lookup(tu.input.get("key", ""))
                else:
                    result = f"unknown tool: {tu.name}"
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tu.id, "content": result}
                )
            self.history.append({"role": "user", "content": tool_results})

        return "[subagent hit max_turns without producing a final reply]"
