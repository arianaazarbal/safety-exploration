import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

try:
    import google.generativeai as genai
except Exception:
    genai = None


class LLMMessage(BaseModel):
    role: str  # system|user|assistant|tool
    content: str


class LLMResponse(BaseModel):
    text: str
    usage_tokens: Optional[int] = None
    raw: Optional[Any] = None


class GeminiClient:
    def __init__(self, model: str = "gemini-1.5-flash", temperature: float = 0.8, top_p: float = 0.95,
                 max_output_tokens: int = 1024, safety_settings: Optional[List[Dict[str, Any]]] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        if genai is None:
            raise RuntimeError("google-generativeai is not installed")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.safety_settings = safety_settings
        self._model = genai.GenerativeModel(model)

    def complete(self, messages: List[LLMMessage]) -> LLMResponse:
        # Compose a single prompt (system first)
        system_prompts = [m.content for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]
        prompt = "\n".join(system_prompts) + ("\n\n" if system_prompts else "") + \
                 "\n".join([f"{m.role.upper()}: {m.content}" for m in non_system])
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_output_tokens": self.max_output_tokens,
                },
                safety_settings=self.safety_settings,
            )
            text = resp.text if hasattr(resp, "text") else str(resp)
            return LLMResponse(text=text, usage_tokens=None, raw=resp)
        except Exception as e:
            return LLMResponse(text=f"<ERROR: {type(e).__name__}: {e}>", usage_tokens=None, raw=None)


def get_client(name: str, **kwargs) -> GeminiClient:
    if name.startswith("gemini"):
        return GeminiClient(model=name, **kwargs)
    raise ValueError(f"Unknown model/client: {name}")
