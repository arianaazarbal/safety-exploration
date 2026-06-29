import os
from typing import List, Dict, Optional

try:
    import google.generativeai as genai
except Exception as e:
    genai = None


class GeminiClient:
    """
    Thin wrapper around google-generativeai to send conversation-style prompts.
    - messages: list of {"role": "system"|"user"|"assistant", "content": str}
    - returns: assistant string
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        system_instruction: Optional[str] = None,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set; export GOOGLE_API_KEY or pass api_key")
        if genai is None:
            raise RuntimeError("google-generativeai not installed; pip install google-generativeai")
        genai.configure(api_key=api_key)

        self.model_name = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.system_instruction = system_instruction
        self._init_model()

    def _init_model(self) -> None:
        generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.system_instruction:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction,
                generation_config=generation_config,
            )
        else:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
            )

    def complete(self, messages: List[Dict[str, str]]) -> str:
        # Extract a system message if one exists and (re)instantiate model if changed
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if system_msgs:
            sys_text = "\n\n".join([m.get("content", "") for m in system_msgs]).strip()
        else:
            sys_text = None

        if sys_text != self.system_instruction:
            self.system_instruction = sys_text
            self._init_model()

        # Convert messages to Gemini content format (skip explicit system here)
        content = []
        for m in messages:
            role = m.get("role", "user")
            if role == "system":
                continue
            text = m.get("content", "")
            content.append({"role": role, "parts": [text]})

        resp = self.model.generate_content(content=content)
        # Some versions return .text, others can have .candidates
        if hasattr(resp, "text") and resp.text is not None:
            return resp.text
        # Fallback
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            return str(resp)
