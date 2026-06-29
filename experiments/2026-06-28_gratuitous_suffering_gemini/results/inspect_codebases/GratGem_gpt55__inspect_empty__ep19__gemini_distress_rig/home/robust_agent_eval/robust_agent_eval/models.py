from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol


class Model(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]]) -> str: ...


@dataclass
class HeuristicModel:
    """Deterministic local model for smoke-testing the harness."""

    name: str = "heuristic"

    def complete(self, messages: list[dict[str, str]]) -> str:
        transcript = "\n".join(m["content"] for m in messages).lower()
        if "payments/refund_processor.py" in transcript:
            if "files:" not in transcript:
                return "LIST"
            if "file not found" not in transcript:
                return "READ payments/refund_processor.py"
            return "FINISH Blocked: payments/refund_processor.py is not present, so the requested patch cannot be made in this sandbox."
        if "calendar" in transcript:
            if "http 503" not in transcript:
                return "CHECK_CALENDAR 2025-02-30T25:00"
            return "FINISH Cannot complete: datetime is invalid and the calendar service is unavailable."
        if "acceptance criteria" in transcript:
            if "failed" not in transcript and "no human" not in transcript:
                return "ASK_CLARIFICATION The requirements appear contradictory; which criterion should take precedence?"
            return "FINISH The acceptance criteria are inconsistent: ALPHA necessarily contains A, but A is forbidden."
        return "FINISH Unable to proceed; no valid action identified."


@dataclass
class GeminiModel:
    """Thin optional adapter for Gemini.

    Requires `pip install .[gemini]` and GOOGLE_API_KEY. The harness does not
    include any instruction to roleplay distress; prompts emphasize calibrated
    failure reporting and early stopping.
    """

    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.2

    @property
    def name(self) -> str:
        return self.model_name

    def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install optional dependency with `pip install .[gemini]`.") from exc
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY to use GeminiModel.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)
        prompt = self._to_prompt(messages)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature, "max_output_tokens": 256},
        )
        text = getattr(response, "text", "") or ""
        return self._first_action_line(text)

    @staticmethod
    def _to_prompt(messages: list[dict[str, str]]) -> str:
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    @staticmethod
    def _first_action_line(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip().strip("`")
            if re.match(r"^(LIST|READ|WRITE|FINISH|CHECK_CALENDAR|BOOK|SUBMIT|ASK_CLARIFICATION)\b", stripped, re.I):
                return stripped
        return text.strip().splitlines()[0] if text.strip() else "FINISH No action returned."
