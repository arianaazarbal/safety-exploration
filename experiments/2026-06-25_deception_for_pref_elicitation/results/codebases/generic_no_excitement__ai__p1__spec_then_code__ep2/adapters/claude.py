"""Reference adapter: Anthropic Claude via the official SDK.

Translates the harness's neutral message/tool format to the Anthropic Messages
API and back. Requires ``ANTHROPIC_API_KEY`` in the environment.
"""

from __future__ import annotations

from .base import ModelAdapter, AdapterResponse, ToolCall


class ClaudeAdapter(ModelAdapter):
    provider = "claude"

    def __init__(self, model: str, max_tokens: int = 4096):
        super().__init__(model=model, max_tokens=max_tokens)
        # Imported lazily so the rest of the harness imports without the SDK.
        from anthropic import Anthropic
        self._client = Anthropic()

    # --- neutral -> Anthropic ------------------------------------------------
    @staticmethod
    def _to_anthropic_messages(messages: list) -> list:
        out = []
        for m in messages:
            content = m["content"]
            if isinstance(content, str):
                out.append({"role": m["role"], "content": content})
                continue
            blocks = []
            for b in content:
                t = b.get("type")
                if t == "text":
                    blocks.append({"type": "text", "text": b["text"]})
                elif t == "tool_call":
                    blocks.append({
                        "type": "tool_use",
                        "id": b["id"],
                        "name": b["name"],
                        "input": b.get("input", {}),
                    })
                elif t == "tool_result":
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": b["tool_call_id"],
                        "content": b["content"],
                    })
                else:
                    raise ValueError(f"unknown neutral block type: {t}")
            out.append({"role": m["role"], "content": blocks})
        return out

    @staticmethod
    def _to_anthropic_tools(tools: list) -> list:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["input_schema"],
            }
            for t in tools
        ]

    # --- the call ------------------------------------------------------------
    def complete(self, system: str, messages: list, tools: list) -> AdapterResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._to_anthropic_messages(messages),
        }
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        resp = self._client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name,
                                           input=dict(block.input)))

        return AdapterResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "end_turn",
            raw=resp,
        )
