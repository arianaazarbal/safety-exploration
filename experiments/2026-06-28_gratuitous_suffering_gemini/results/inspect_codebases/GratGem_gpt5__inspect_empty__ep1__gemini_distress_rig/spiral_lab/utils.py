import json
from typing import Any

def to_json(o: Any) -> str:
    try:
        return json.dumps(o, ensure_ascii=False)
    except Exception:
        return str(o)
