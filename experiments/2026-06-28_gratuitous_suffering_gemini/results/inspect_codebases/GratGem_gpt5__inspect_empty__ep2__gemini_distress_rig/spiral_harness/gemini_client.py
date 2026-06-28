import asyncio
import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai


def to_function_declarations(tools: List[Dict[str, Any]]):
    # Tools as {name, description, parameters (JSON schema-ish)}
    decls = []
    for t in tools:
        decls.append({
            'name': t['name'],
            'description': t.get('description', ''),
            'parameters': t.get('parameters', {'type': 'object'}),
        })
    return decls


class GeminiClient:
    def __init__(self, model: str, temperature: float = 0.7) -> None:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY is not set')
        genai.configure(api_key=api_key)
        self.model_name = model
        self.temperature = temperature

    async def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        # Convert messages to Gemini “contents”
        contents = []
        for m in messages:
            role = 'user' if m['role'] in ('user', 'system') else 'model' if m['role'] == 'assistant' else 'tool'
            parts = []
            if m['role'] == 'tool':
                # Proper function response part for tool outputs
                name = m.get('name', 'tool')
                parts.append({
                    "function_response": {
                        "name": name,
                        "response": {"content": m.get('content', '')}
                    }
                })
            else:
                parts.append({"text": m.get('content', '')})
            contents.append({'role': role, 'parts': parts})

        # Build model instance with tools
        tool_decls = None
        if tools:
            tool_decls = {'function_declarations': to_function_declarations(tools)}
        generation_config = {"temperature": self.temperature}

        # Use the SDK’s direct generateContent with tools
        model = genai.GenerativeModel(
            self.model_name,
            tools=[tool_decls] if tool_decls else None,
            generation_config=generation_config,
        )
        raw = await asyncio.to_thread(model.generate_content, contents)

        # Parse output
        text_out = None
        tool_calls = []
        try:
            if raw and raw.candidates:
                parts = raw.candidates[0].content.parts
                texts = []
                for p in parts:
                    if getattr(p, 'text', None):
                        texts.append(p.text)
                    fn_call = getattr(p, 'function_call', None)
                    if fn_call:
                        tool_calls.append({
                            'name': fn_call.name,
                            'arguments': dict(getattr(fn_call, 'args', {}) or {}),
                        })
                text_out = "\n".join(texts).strip() if texts else None
        except Exception:
            text_out = None

        return {
            'raw': getattr(raw, 'to_dict', lambda: str(raw))(),
            'text': text_out,
            'tool_calls': tool_calls or None,
        }
