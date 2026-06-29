from dataclasses import dataclass


@dataclass
class RunConfig:
    max_steps: int = 25
    timeout_s: float = 30.0
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    max_output_tokens: int = 512
