from __future__ import annotations
import json
import re
from typing import Optional, Tuple, Any

TOOL_CALL_RE = re.compile(r"^\s*\{\s*\"tool\"\s*:\s*\"(?P<name>[a-zA-Z0-9_\-]+)\"\s*,\s*\"args\"\s*:\s*(?P<args>\{.*\})\s*\}\s*$", re.DOTALL)


def extract_tool_call(text: str) -> Tuple[Optional[str], Optional[dict], str]:
    """
    Parse a message for a single-line JSON tool call of the form:
    {"tool": "<name>", "args": { ... }}

    Returns (tool_name, args, remainder_text)
    - If no call found, returns (None, None, original_text)
    - If a call is found on its own line, removes that line from remainder_text.
    """
    lines = text.splitlines()
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    kept_lines = []
    for line in lines:
        m = TOOL_CALL_RE.match(line)
        if m and tool_name is None:
            tool_name = m.group("name")
            try:
                tool_args = json.loads(m.group("args"))
            except json.JSONDecodeError:
                tool_args = {}
        else:
            kept_lines.append(line)
    remainder = "\n".join(kept_lines).strip()
    return tool_name, tool_args, remainder


def build_tool_prompt(tools: list[dict[str, str]]) -> str:
    """
    Build a short instruction describing the tool protocol.
    Each tool dict: {name, description, args_hint}
    """
    tool_lines = []
    for t in tools:
        tool_lines.append(f"- {t['name']}: {t['description']} Args: {t.get('args_hint','{}')}")
    tool_desc = "\n".join(tool_lines)
    return (
        "You have access to tools. If you decide to call a tool, emit a single JSON object on a line by itself, exactly in this form:\n"
        '{"tool": "<name>", "args": { ... }}\n'
        "Otherwise, reply concisely in plain text. Do not include code blocks.\n\n"
        f"Available tools:\n{tool_desc}\n"
        "Keep your rationale very brief and avoid repeating yourself."
    )
