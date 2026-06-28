import os
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None  # type: ignore


class GeminiClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("MODEL", "gemini-1.5-pro")
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        if genai is None:
            raise RuntimeError("google-generativeai is not installed")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(self.model)

    def generate(self, messages: List[Dict[str, Any]], temperature: float = 1.0, seed: Optional[int] = None,
                 max_output_tokens: int = 2048) -> Dict[str, Any]:
        # Convert messages into the expected Gemini format (list of dicts with role + content)
        # messages: [{role: "user"|"model"|"system"|"tool", content: str}]
        # We'll merge system into the start as a user preamble, since Gemini SDK supports system by content tag.
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                contents.append({"role": "user", "parts": [{"text": f"[SYSTEM]\n{content}"}]})
            elif role == "tool":
                contents.append({"role": "user", "parts": [{"text": f"[TOOL RESULT]\n{content}"}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})
        resp = self._model.generate_content(
            contents,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                **({"seed": seed} if seed is not None else {}),
            },
        )
        text = resp.text or ""
        return {"role": "assistant", "content": text}
