from __future__ import annotations
import os
from typing import Sequence

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class GeminiClient:
    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 1.0, top_p: float = 0.95, max_output_tokens: int = 512):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
        }

    @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=20), retry=retry_if_exception_type(Exception))
    def generate(self, messages: Sequence[dict]) -> str:
        # messages: list of {role: "system"|"user"|"assistant", content: str}
        # For Gemini, we can convert to a Chat session
        # We'll flatten into a single prompt by alternating roles
        chat = self.model.start_chat(history=[])
        # Apply system instructions by prepending to first user turn.
        # If multiple system messages, concat them.
        sys_prompt = "\n\n".join([m["content"] for m in messages if m["role"] == "system"]).strip()
        for m in messages:
            if m["role"] == "user":
                content = m["content"]
                if sys_prompt:
                    content = sys_prompt + "\n\n" + content
                    sys_prompt = ""  # only prepend once
                chat.send_message(content, generation_config=self.generation_config)
            elif m["role"] == "assistant":
                chat.send_message(m["content"], generation_config=self.generation_config)
        resp = chat.send_message("", generation_config=self.generation_config)  # prompt with empty to get next turn
        text = resp.text or ""
        return text.strip()
