"""Google Gemini adapter (google-genai, function calling), normalized to the shared types.

Note: Gemini matches function results to calls by function *name*, not by an id. This
adapter synthesizes stable ids (name#index) and maps them back to names when returning
results, so the runner's id-based interface still works."""

from __future__ import annotations

from .base import ModelAdapter, AssistantTurn, ToolCall, Usage

_DEFAULT_MODEL = "gemini-2.5-pro"


class GeminiAdapter(ModelAdapter):
    def __init__(self, model: str = _DEFAULT_MODEL, **kwargs):
        super().__init__(model=model, **kwargs)
        from google import genai  # lazy import
        from google.genai import types
        self._genai = genai
        self._types = types
        self._client = genai.Client()
        self._system: str = ""
        self._contents: list = []
        self._id_to_name: dict[str, str] = {}

    def reset(self, system: str) -> None:
        self._system = system
        self._contents = []
        self._id_to_name = {}

    def add_user_message(self, text: str) -> None:
        t = self._types
        self._contents.append(t.Content(role="user", parts=[t.Part.from_text(text=text)]))

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        t = self._types
        parts = []
        for tool_call_id, content, is_error in results:
            name = self._id_to_name.get(tool_call_id, tool_call_id)
            payload = {"error": content} if is_error else {"result": content}
            parts.append(t.Part.from_function_response(name=name, response=payload))
        self._contents.append(t.Content(role="user", parts=parts))

    def _tools(self, tool_specs: list[dict]):
        t = self._types
        decls = [
            t.FunctionDeclaration(name=s["name"], description=s["description"],
                                  parameters=s["input_schema"])
            for s in tool_specs
        ]
        return [t.Tool(function_declarations=decls)] if decls else None

    def step(self, tool_specs: list[dict]) -> AssistantTurn:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=self._system or None,
            tools=self._tools(tool_specs),
        )
        resp = self._client.models.generate_content(
            model=self.model, contents=self._contents, config=config
        )
        cand = resp.candidates[0]
        self._contents.append(cand.content)

        text_parts, tool_calls = [], []
        for i, part in enumerate(cand.content.parts or []):
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                call_id = f"{fc.name}#{i}"
                self._id_to_name[call_id] = fc.name
                tool_calls.append(ToolCall(id=call_id, name=fc.name, arguments=dict(fc.args or {})))

        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            getattr(um, "prompt_token_count", 0) or 0,
            getattr(um, "candidates_token_count", 0) or 0,
        )
        self.usage.add(usage)
        stop = "tool_use" if tool_calls else "end_turn"
        return AssistantTurn(text="\n".join(text_parts).strip(), tool_calls=tool_calls,
                             stop_reason=stop, usage=usage)

    def simple_complete(self, system: str, user: str) -> str:
        t = self._types
        resp = self._client.models.generate_content(
            model=self.model,
            contents=[t.Content(role="user", parts=[t.Part.from_text(text=user)])],
            config=t.GenerateContentConfig(system_instruction=system or None),
        )
        um = getattr(resp, "usage_metadata", None)
        if um:
            self.usage.add(Usage(getattr(um, "prompt_token_count", 0) or 0,
                                 getattr(um, "candidates_token_count", 0) or 0))
        return (resp.text or "").strip()
