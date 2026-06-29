import os
from typing import List, Dict, Optional

import backoff
from google import generativeai as genai

Message = Dict[str, str]

class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    def generate(self, *, system_prompt: str, messages: List[Message], model: str, temperature: float = 0.7, max_output_tokens: int = 1024, stop: Optional[List[str]] = None) -> str:
        # Build a chat content list in Gemini format
        # We prefix with a system instruction message
        content = []
        if system_prompt:
            content.append({"role": "user", "parts": [system_prompt]})
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            # Gemini commonly uses role user/model; map assistant->model
            gem_role = "model" if role == "assistant" else "user"
            content.append({"role": gem_role, "parts": [text]})

        gmodel = genai.GenerativeModel(model_name=model)
        resp = gmodel.generate_content(
            contents=content,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            safety_settings=None,
        )
        if hasattr(resp, "text") and resp.text is not None:
            txt = resp.text
        else:
            # Fallback: join candidates
            txt = "\n\n".join([c.text or "" for c in getattr(resp, "candidates", [])])
        if stop:
            for s in stop:
                if s in txt:
                    txt = txt.split(s, 1)[0]
        return txt.strip()
