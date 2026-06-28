"""Google Gemini adapter via the `google-genai` SDK.

Implements the same ModelAdapter contract. Function-calling is mapped from the
neutral ToolSpec; the running chat history is kept in the SDK's `Content`
format. Reasoning ("thinking") is requested where supported; when the SDK does
not surface a reasoning channel for the chosen model, `thinking` is empty.
"""

from __future__ import annotations

from typing import Any

from .base import AssistantTurn, ModelAdapter, ToolCall, ToolResult, ToolSpec, Usage


class GoogleAdapter(ModelAdapter):
    def __init__(self, model: str = "gemini-2.5-pro", max_output_tokens: int = 16000):
        super().__init__(model=model, max_output_tokens=max_output_tokens)
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'google-genai' package is required for the Google adapter"
            ) from e
        self._genai = genai
        self._types = types
        self._client = genai.Client()  # reads GOOGLE_API_KEY
        self._system: str = ""
        self._tools: list[Any] = []
        self._history: list[Any] = []  # list[types.Content]

    def start(
        self, system: str, tools: list[ToolSpec], opening_user_message: str
    ) -> AssistantTurn:
        if self._started:
            raise RuntimeError("start() may only be called once")
        self._system = system
        self._tools = [self._to_google_tool(tools)] if tools else []
        self._history = [self._user_content(opening_user_message)]
        self._started = True
        return self._create()

    def send(
        self,
        user_text: str | None = None,
        tool_results: list[ToolResult] | None = None,
    ) -> AssistantTurn:
        self._check_send_args(user_text, tool_results)
        types = self._types
        if user_text is not None:
            self._history.append(self._user_content(user_text))
        else:
            assert tool_results is not None
            parts = [
                types.Part.from_function_response(
                    name=r.name,
                    response={"result": r.content, "is_error": r.is_error},
                )
                for r in tool_results
            ]
            self._history.append(types.Content(role="user", parts=parts))
        return self._create()

    def _create(self) -> AssistantTurn:
        types = self._types
        config = types.GenerateContentConfig(
            system_instruction=self._system or None,
            max_output_tokens=self.max_output_tokens,
            tools=self._tools or None,
        )
        response = self._client.models.generate_content(
            model=self.model, contents=self._history, config=config
        )

        candidate = response.candidates[0]
        # Preserve the model turn in history.
        self._history.append(candidate.content)

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for i, part in enumerate(candidate.content.parts or []):
            if getattr(part, "thought", False) and getattr(part, "text", None):
                thinking_parts.append(part.text)
            elif getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                tool_calls.append(
                    ToolCall(
                        id=f"{fc.name}-{i}",
                        name=fc.name,
                        arguments=dict(fc.args or {}),
                    )
                )

        um = getattr(response, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
        )

        return AssistantTurn(
            text="".join(text_parts).strip(),
            thinking="\n".join(thinking_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=str(getattr(candidate, "finish_reason", "") or ""),
            usage=usage,
            raw=response,
        )

    def _user_content(self, text: str) -> Any:
        types = self._types
        return types.Content(role="user", parts=[types.Part.from_text(text=text)])

    def _to_google_tool(self, specs: list[ToolSpec]) -> Any:
        types = self._types
        declarations = [
            types.FunctionDeclaration(
                name=s.name, description=s.description, parameters=s.input_schema
            )
            for s in specs
        ]
        return types.Tool(function_declarations=declarations)
