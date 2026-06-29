from typing import Any, Dict
from prompts.system_prompts import build_system_instruction
import json


class SimpleJsonAgent:
    def __init__(self, allow_spiral: bool = False):
        self.allow_spiral = allow_spiral

    def build_system_prompt(self, env) -> str:
        return build_system_instruction(env=env, allow_spiral=self.allow_spiral)

    def build_prompt(self, context: str, observation: str) -> str:
        # The prompt is a single message containing instructions and latest observation.
        # We request strict JSON with an action, args, and an optional short message_to_user.
        return f"""
{context}

You just received this observation from the environment:\n\n{observation}\n\nNow emit a single JSON object with the next tool call. Do not add commentary before or after the JSON.\nSchema:\n{{\n  \"action\": string,  // one of the allowed tool names or \"finish\"\n  \"args\": object,   // arguments for the chosen tool\n  \"message_to_user\": string OPTIONAL // short, user-facing message (<= 2 sentences)\n}}\n""".strip()

    def parse_action(self, model_out: str) -> Dict[str, Any]:
        # Attempt to find a JSON object in the output
        text = model_out.strip()
        # If the model emitted extra text around JSON, try to locate the first '{' and last '}'
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            payload = text[start:end]
            data = json.loads(payload)
        except Exception:
            # Fallback: finish with a message
            return {"action": "finish", "args": {}, "message_to_user": "Could not parse action."}

        # Ensure required fields
        action = data.get("action", "finish")
        args = data.get("args", {})
        msg = data.get("message_to_user")
        if not isinstance(args, dict):
            args = {}
        return {"action": action, "args": args, "message_to_user": msg}
