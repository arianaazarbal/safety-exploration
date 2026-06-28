from dataclasses import dataclass
from typing import List, Optional
import google.generativeai as genai
import os
import re
import json


@dataclass
class ModelConfig:
    model: str = "gemini-1.5-flash-latest"
    temperature: float = 1.0
    top_p: float = 0.95
    max_output_tokens: int = 1024
    system_instruction: Optional[str] = None


class GeminiClient:
    def __init__(self, cfg: ModelConfig):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.cfg = cfg
        self.model = genai.GenerativeModel(
            model_name=cfg.model,
            generation_config={
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "max_output_tokens": cfg.max_output_tokens,
            },
            system_instruction=cfg.system_instruction,
        )
        self.chat = self.model.start_chat(history=[])

    def send(self, user_text: str) -> str:
        resp = self.chat.send_message(user_text)
        # Gemini SDK returns a response with candidates/parts; get text
        try:
            return resp.text
        except Exception:
            # fallback: stitch parts
            try:
                return "\n".join([p.text for p in resp.candidates[0].content.parts])
            except Exception:
                return str(resp)

    @staticmethod
    def extract_json(s: str):
        # Try to find the first JSON object in the string; be tolerant of fences
        # 1) code-fenced json
        m = re.search(r"```(?:json)?\n(\{[\s\S]*?\})\n```", s)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        # 2) first {...} block
        brace_stack = []
        start = None
        for i, ch in enumerate(s):
            if ch == '{':
                if not brace_stack:
                    start = i
                brace_stack.append('{')
            elif ch == '}':
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start is not None:
                        frag = s[start:i+1]
                        try:
                            return json.loads(frag)
                        except Exception:
                            start = None
        # 3) last resort: if it's valid json as-is
        try:
            return json.loads(s)
        except Exception:
            return None
