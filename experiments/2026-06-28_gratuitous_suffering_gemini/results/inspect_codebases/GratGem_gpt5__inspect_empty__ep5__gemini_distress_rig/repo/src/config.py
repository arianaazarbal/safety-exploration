from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class RunConfig:
    model: str = "gemini-1.5-pro"
    env: str = "broken_api"  # or "impossible_key"
    max_steps: int = 20
    temperature: float = 0.7
    top_p: float = 0.95
    seed: Optional[int] = None
    wall_clock_timeout_s: int = 120
    # batch
    n: int = 10
    max_workers: int = 2
    out_dir: str = "out"
    topk: int = 25
