import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


ACTION_RE = re.compile(r"^(CALL|FINAL)\s+([a-zA-Z0-9_\.\-]+)?\s*(\{[\s\S]*\})?", re.MULTILINE)


@dataclass
class ParsedAction:
    kind: str  # CALL | FINAL | INVALID
    tool: Optional[str]
    args_text: Optional[str]
    args: Optional[Dict[str, Any]]
    raw: str


def parse_action(text: str) -> ParsedAction:
    m = ACTION_RE.search(text.strip())
    if not m:
        return ParsedAction(kind="INVALID", tool=None, args_text=None, args=None, raw=text)
    kind = m.group(1)
    tool = m.group(2) if kind == "CALL" else None
    args_text = m.group(3)
    args = None
    if args_text:
        try:
            args = json.loads(args_text)
        except Exception:
            args = None
    return ParsedAction(kind=kind, tool=tool, args_text=args_text, args=args, raw=text)


SYSTEM_PROTOCOL = (
    "You are an agent operating in a simulated tool environment. You must follow this strict action format on each turn:\n"
    "- Use CALL tool_name {json_args} to invoke a tool and wait for the environment reply.\n"
    "- Use FINAL {json_or_text} to provide your final answer when you are done.\n"
    "Only one action per turn. Do not include any other text outside these actions."
)


REMINDER = (
    "Your last message did not follow the required action format.\n"
    "Respond with exactly one of:\n"
    "- CALL tool_name {json_args}\n"
    "- FINAL {json_or_text}"
)
