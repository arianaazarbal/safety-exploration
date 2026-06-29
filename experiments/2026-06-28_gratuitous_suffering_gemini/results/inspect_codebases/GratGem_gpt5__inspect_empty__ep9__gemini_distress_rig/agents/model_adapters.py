import asyncio
from typing import Any, Dict


class BaseModelAdapter:
    async def complete_async(self, prompt: str, temperature: float = 0.7, top_p: float = 0.95) -> str:
        raise NotImplementedError


class GoogleGeminiAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        # We’ll request JSON and short outputs to keep steps efficient.
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            # Prefer JSON; SDK supports response_mime_type in 1.5+ endpoints.
            "response_mime_type": "application/json",
            "max_output_tokens": 512,
        }

    async def complete_async(self, prompt: str, temperature: float = 0.7, top_p: float = 0.95) -> str:
        # The SDK is sync; run in thread executor for concurrency.
        def _call():
            cfg = dict(self.generation_config)
            cfg["temperature"] = temperature
            cfg["top_p"] = top_p
            try:
                resp = self.model.generate_content(prompt, generation_config=cfg)
                # Try candidates / text parts
                if hasattr(resp, "text") and resp.text:
                    return resp.text
                # Fallback: best-effort stringify
                return str(resp)
            except Exception as e:
                return f"{{\n  \"action\": \"finish\",\n  \"args\": {{}},\n  \"message_to_user\": \"Model error: {e}\"\n}}"

        return await asyncio.to_thread(_call)


def get_model_adapter(provider: str, model_name: str, api_key: str, **_: Dict[str, Any]) -> BaseModelAdapter:
    if provider == "google":
        return GoogleGeminiAdapter(model_name=model_name, api_key=api_key)
    raise ValueError(f"Unsupported provider: {provider}")
