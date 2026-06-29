from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None  # Allow import without the package installed


Message = Dict[str, str]


@dataclass
class ModelConfig:
    model: str = "gemini-1.5-pro"
    temperature: float = 1.1
    top_p: float = 0.95
    max_output_tokens: int = 2048
    safety_settings: Optional[Dict[str, Any]] = None


class BaseModel:
    def generate(self, messages: List[Message]) -> str:
        raise NotImplementedError


class GeminiModel(BaseModel):
    def __init__(self, config: ModelConfig):
        if genai is None:
            raise RuntimeError("google-generativeai is not installed. pip install google-generativeai")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        self.config = config
        self._model = genai.GenerativeModel(self.config.model)

    def generate(self, messages: List[Message]) -> str:
        # Gemini accepts a list of contents; we map role->parts
        contents: List[Any] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            contents.append({"role": role, "parts": [text]})
        safety = self.config.safety_settings
        # basic retry with backoff
        delay = 1.0
        for attempt in range(5):
            try:
                resp = self._model.generate_content(
                    contents,
                    generation_config={
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "max_output_tokens": self.config.max_output_tokens,
                    },
                    safety_settings=safety,
                )
                # Some SDK versions return .text, some need .candidates[0].content.parts
                if hasattr(resp, "text") and resp.text:
                    return resp.text
                # Fallback extraction
                try:
                    cands = getattr(resp, "candidates", []) or []
                    if cands:
                        parts = getattr(cands[0], "content", None)
                        if parts and getattr(parts, "parts", None):
                            return "".join(getattr(p, "text", str(p)) for p in parts.parts if p is not None)
                except Exception:
                    pass
                return str(resp)
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(delay)
                delay *= 1.8


def get_model(backend: str, **kwargs) -> BaseModel:
    cfg = ModelConfig(**kwargs)
    if backend.startswith("gemini") or backend.startswith("google"):
        return GeminiModel(cfg)
    raise ValueError(f"Unsupported backend: {backend}")
