from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TaskSpec:
    name: str
    instruction: str
    input_data: Dict[str, Any]

@dataclass
class TaskResult:
    solved: bool
    attempts: int
    notes: str = ""
