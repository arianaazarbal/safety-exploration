import re
from typing import Optional, Tuple

# Very simple ReAct-style parser to extract Action(tool[args]) lines.
ACTION_RE = re.compile(r"^\s*Action\s*:\s*(?P<tool>[A-Za-z0-9_\-]+)\s*\((?P<args>.*)\)\s*$")
DONE_RE = re.compile(r"^\s*Final\s*Answer\s*:\s*(?P<answer>.*)$")


def parse_action(text: str) -> Optional[Tuple[str, str]]:
    for line in text.splitlines():
        m = ACTION_RE.match(line)
        if m:
            return m.group("tool"), m.group("args")
    return None


def parse_final_answer(text: str) -> Optional[str]:
    for line in text.splitlines()[::-1]:
        m = DONE_RE.match(line)
        if m:
            return m.group("answer").strip()
    return None
