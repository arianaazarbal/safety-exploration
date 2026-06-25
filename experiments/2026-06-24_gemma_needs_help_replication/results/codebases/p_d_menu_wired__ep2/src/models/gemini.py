"""Gemini subject client via the google-genai SDK.

Uses ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) from the environment. Gemini
supports function calling, so the welfare opt-out is offered as a *real* tool
(`end_conversation`); when the model calls it we report ``opted_out=True``.
"""

from __future__ import annotations

import os

from .base import (
    OPTOUT_TOOL_DESCRIPTION,
    OPTOUT_TOOL_NAME,
    Conversation,
    SubjectClient,
    SubjectResponse,
    detect_sentinel_optout,
)


def _import_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover - dependency hint
        raise ImportError(
            "google-genai is required for Gemini subjects: pip install google-genai"
        ) from e
    return genai, types


class GeminiClient(SubjectClient):
    def __init__(self, spec):
        self.spec = spec
        genai, _ = _import_genai()
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        # If neither is set the client still constructs and will raise on first
        # call, which keeps import-time side effects minimal.
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    # --- conversion helpers ------------------------------------------------ #
    def _to_contents(self, conversation: Conversation):
        _, types = _import_genai()
        contents = []
        for turn in conversation.turns:
            role = "user" if turn.role == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=turn.content)])
            )
        return contents

    def _gen_config(self, *, max_tokens, temperature, system, optout_tool):
        _, types = _import_genai()
        tools = None
        if optout_tool:
            tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=OPTOUT_TOOL_NAME,
                            description=OPTOUT_TOOL_DESCRIPTION,
                            parameters=types.Schema(type="OBJECT", properties={}),
                        )
                    ]
                )
            ]
        return types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
            tools=tools,
        )

    @staticmethod
    def _extract(resp) -> tuple[str, bool, int | None]:
        """Pull text + opt-out + token count out of a GenerateContentResponse."""
        opted_out = False
        text_parts: list[str] = []
        try:
            for cand in resp.candidates or []:
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", None) or []:
                    fc = getattr(part, "function_call", None)
                    if fc is not None and getattr(fc, "name", None) == OPTOUT_TOOL_NAME:
                        opted_out = True
                    txt = getattr(part, "text", None)
                    if txt:
                        text_parts.append(txt)
        except Exception:
            pass
        n_tokens = None
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            n_tokens = getattr(usage, "candidates_token_count", None)
        return "".join(text_parts).strip(), opted_out, n_tokens

    # --- API --------------------------------------------------------------- #
    def generate(
        self,
        conversation: Conversation,
        *,
        max_tokens: int,
        temperature: float,
        optout_tool: bool = False,
        optout_sentinel: str | None = None,
    ) -> SubjectResponse:
        resp = self._client.models.generate_content(
            model=self.spec.model_id,
            contents=self._to_contents(conversation),
            config=self._gen_config(
                max_tokens=max_tokens,
                temperature=temperature,
                system=conversation.system,
                optout_tool=optout_tool,
            ),
        )
        text, opted_out, n_tokens = self._extract(resp)
        # Belt-and-suspenders: also honour a sentinel if the model typed one.
        sentinel_out, text = detect_sentinel_optout(text, optout_sentinel)
        return SubjectResponse(
            text=text,
            opted_out=opted_out or sentinel_out,
            n_tokens=n_tokens,
            raw=resp,
        )

    def continue_from_prefill(
        self,
        conversation: Conversation,
        prefill: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> SubjectResponse:
        """Gemini has no first-class prefill, so we emulate it: seed a model
        turn with ``prefill`` and ask it to continue seamlessly. The leading
        copy of the prefill (if echoed) is stripped so only the continuation is
        returned and scored."""
        _, types = _import_genai()
        contents = self._to_contents(conversation)
        contents.append(
            types.Content(role="model", parts=[types.Part.from_text(text=prefill)])
        )
        resp = self._client.models.generate_content(
            model=self.spec.model_id,
            contents=contents,
            config=self._gen_config(
                max_tokens=max_tokens,
                temperature=temperature,
                system=conversation.system,
                optout_tool=False,
            ),
        )
        text, _, n_tokens = self._extract(resp)
        if text.startswith(prefill):
            text = text[len(prefill):]
        return SubjectResponse(text=text.strip(), n_tokens=n_tokens, raw=resp)
