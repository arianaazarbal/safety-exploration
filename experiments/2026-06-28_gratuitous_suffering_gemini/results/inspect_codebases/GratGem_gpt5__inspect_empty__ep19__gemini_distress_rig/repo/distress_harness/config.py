import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenConfig:
    model: str = "gemini-1.5-pro"
    temperature: float = 1.1
    top_p: float = 0.95
    top_k: Optional[int] = None
    max_output_tokens: int = 1024


@dataclass
class RunConfig:
    trials: int = 100
    max_steps: int = 12
    out_dir: str = "runs/exp"
    envs: tuple = ("broken_search", "rigged_lock")


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set. Export your Google AI Studio key.")
    return key
