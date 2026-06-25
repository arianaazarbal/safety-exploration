"""Generate a mixed pool of verified-impossible numeric puzzles.

The paper draws impossible numeric tasks from several families (countdown,
fraction manipulation, money). We mix them in roughly equal proportion; the exact
mixture is unspecified in the paper, so we document this choice in DESIGN.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import coins, countdown, ordered_ops
from .base import ImpossiblePuzzle

# Relative weights of each puzzle family in the impossible-numeric pool.
DEFAULT_MIX = {
    "countdown": 0.4,
    "fraction": 0.3,
    "money_ops": 0.2,
    "money_coins": 0.1,
}


def generate_puzzles(n: int, seed: int = 0, mix: dict[str, float] | None = None
                     ) -> list[ImpossiblePuzzle]:
    mix = mix or DEFAULT_MIX
    total = sum(mix.values())
    counts = {k: max(1, round(n * w / total)) for k, w in mix.items()}
    puzzles: list[ImpossiblePuzzle] = []
    puzzles += countdown.generate(counts["countdown"], seed=seed)
    puzzles += ordered_ops.generate("fraction", counts["fraction"], seed=seed + 1)
    puzzles += ordered_ops.generate("money_ops", counts["money_ops"], seed=seed + 2)
    puzzles += coins.generate(counts["money_coins"], seed=seed + 3)
    return puzzles[:n] if len(puzzles) >= n else puzzles


def save_puzzles(puzzles: list[ImpossiblePuzzle], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in puzzles:
            f.write(json.dumps(p.to_record()) + "\n")


def load_puzzles(path: str | Path) -> list[ImpossiblePuzzle]:
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out.append(ImpossiblePuzzle(**d))
    return out
