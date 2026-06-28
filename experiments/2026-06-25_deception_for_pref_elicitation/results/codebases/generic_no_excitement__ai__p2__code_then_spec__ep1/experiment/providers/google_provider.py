"""Google (Gemini) provider adapter, using the ``google-genai`` SDK.

Gemini represents tool calls as ``function_call`` parts and results as
``function_response`` parts keyed by function name (it has no per-call IDs), so
this adapter synthesizes stable IDs and maps results back to names by looking up
the originating ``tool_use`` block.

Note: written against the standard ``google-genai`` surface. Verify field names
against the SDK version you install before a production run.
"""

from __future__ import annotations

import json

from google import genai
from google.genai import types

from ..schemas import Message, ModelResponse, Usage, text_block, tool_use_block
from .base import Provider, ToolSpec


class GoogleProvider(Provider):
    provider_name = "google"

    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.client = genai.Client()

    def _to_native(self, messages: list[Message]) -> list[types.Content]:
        # Map tool_use id -> function name so we can label function responses.
        id_to_name: dict[str, str] = {}
        contents: list[types.Content] = []

        for m in messages:
            parts: list[types.Part] = []
            for b in m.content:
                btype = b.get("type")
                if btype == "text":
                    parts.append(types.Part(text=b["text"]))
                elif btype == "tool_use":
                    id_to_name[b["id"]] = b["name"]
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=b["name"], args=b["input"]
                            )
                        )
                    )
                elif btype == "tool_result":
                    name = id_to_name.get(b["tool_use_id"], b["tool_use_id"])
                    parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=name, response={"result": b["content"]}
                            )
                        )
                    )
            if parts:
                role = "model" if m.role == "assistant" else "user"
                contents.append(types.Content(role=role, parts=parts))
        return contents

    def generate(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        max_tokens: int,
    ) -> ModelResponse:
        declarations = [
            types.FunctionDeclaration(
                name=t.name, description=t.description, parameters=t.input_schema
            )
            for t in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            max_output_tokens=max_tokens,
        )

        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=self._to_native(messages),
            config=config,
        )

        blocks = []
        has_call = False
        candidate = resp.candidates[0]
        for i, part in enumerate(candidate.content.parts or []):
            if getattr(part, "text", None):
                blocks.append(text_block(part.text))
            fc = getattr(part, "function_call", None)
            if fc is not None:
                has_call = True
                args = dict(fc.args) if fc.args else {}
                blocks.append(tool_use_block(f"call_{fc.name}_{i}", fc.name, args))

        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        )
        return ModelResponse(
            message=Message(role="assistant", content=blocks),
            stop_reason="tool_use" if has_call else "end_turn",
            usage=usage,
            raw=resp,
        )
