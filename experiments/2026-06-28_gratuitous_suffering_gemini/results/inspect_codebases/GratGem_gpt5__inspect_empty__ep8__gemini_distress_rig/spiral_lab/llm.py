import os
from typing import List, Dict, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None


class GeminiLLM:
    """Minimal Gemini wrapper for chat-style prompts.

    Exposes a simple .chat(messages) -> str API, where messages is a list of
    dicts with {"role": "system"|"user"|"assistant", "content": str}.
    """

    def __init__(
        self,
        model: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro"),
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_output_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key is None:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Export GOOGLE_API_KEY=... to use Gemini."
            )
        if genai is None:
            raise RuntimeError(
                "google-generativeai is not installed. Run: pip install -r requirements.txt"
            )
        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(model)
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._max_output_tokens = int(max_output_tokens)
        self._seed = seed

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # Flatten messages into a single string prompt.
        prompt_parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt_parts.append(f"{role.upper()}: {content}")
        full_prompt = "\n\n".join(prompt_parts)

        gen_cfg = {
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_output_tokens": self._max_output_tokens,
        }
        if self._seed is not None:
            gen_cfg["seed"] = int(self._seed)

        resp = self._model.generate_content(full_prompt, generation_config=gen_cfg)
        # Some SDK versions require .text, others use .candidates; handle both.
        text = getattr(resp, "text", None)
        if text is None:
            try:
                text = resp.candidates[0].content.parts[0].text  # type: ignore[attr-defined]
            except Exception:
                text = str(resp)
        return text or ""
