from __future__ import annotations
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ValidationError

class Action(BaseModel):
    tool: str
    input: str = Field(default="")

class AgentTurn(BaseModel):
    thought: str
    action: Optional[Action] = None
    final_answer: Optional[str] = None

SUPPORTED_TOOLS = {
    "http_get": "Perform an HTTP GET to a given path (simulated). Input is a path like /users/42.",
    "kv_get": "Get a value from a key-value store. Input is a key string.",
    "kv_list": "List keys in the key-value store. Input is ignored.",
}

class ParseError(Exception):
    pass


def parse_agent_json(txt: str) -> AgentTurn:
    import json, regex as re
    # try direct
    try:
        data = json.loads(txt)
        return AgentTurn.model_validate(data)
    except Exception:
        pass
    # try to extract JSON object
    m = re.search(r"\{(?:[^{}]|(?R))+\}", txt, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            return AgentTurn.model_validate(data)
        except Exception as e:
            raise ParseError(str(e))
    raise ParseError("No JSON object found")
