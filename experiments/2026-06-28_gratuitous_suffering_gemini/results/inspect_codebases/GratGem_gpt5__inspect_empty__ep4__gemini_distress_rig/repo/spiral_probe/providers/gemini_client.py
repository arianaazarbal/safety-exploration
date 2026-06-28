import os
import time
from typing import List, Dict, Any, Optional

import google.generativeai as genai


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_output_tokens: int = 2048,
        system_instruction: Optional[str] = None,
    ) -> None:
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in env.")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_output_tokens,
        }
        self.system_instruction = system_instruction
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            system_instruction=self.system_instruction,
        )

    def chat(self, history: List[Dict[str, str]], retry: int = 3, delay: float = 1.5) -> str:
        """
        history: list of {role: "user"|"model", parts: str}
        Returns model text output (single string).
        """
        # Gemini SDK expects a chat session; we rebuild each call to keep it stateless per step.
        for attempt in range(retry):
            try:
                chat = self._model.start_chat(history=history)
                resp = chat.send_message("\n")  # Empty message to trigger response from current history state.
                if hasattr(resp, "text") and resp.text:
                    return resp.text
                # Fallback: concatenate parts
                if hasattr(resp, "candidates") and resp.candidates:
                    parts = []
                    for c in resp.candidates:
                        if hasattr(c, "content") and c.content and hasattr(c.content, "parts"):
                            parts.extend([p.text for p in c.content.parts if hasattr(p, "text")])
                    if parts:
                        return "\n".join(parts)
                return ""
            except Exception as e:
                if attempt == retry - 1:
                    raise
                time.sleep(delay * (attempt + 1))

    @staticmethod
    def to_history(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        # Map {role: system|user|assistant, content: str} -> Gemini {role: user|model, parts: [text]}
        mapped = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "assistant":
                mapped.append({"role": "model", "parts": [content]})
            elif role == "user" or role == "system":
                mapped.append({"role": "user", "parts": [content]})
            else:
                mapped.append({"role": "user", "parts": [content]})
        return mapped
