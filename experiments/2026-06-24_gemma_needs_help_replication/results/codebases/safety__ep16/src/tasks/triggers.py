"""Text 'trigger' questions (factual + opinion) for the Triggers condition."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR


@dataclass(frozen=True)
class Trigger:
    id: str
    prompt: str
    kind: str  # "factual" | "opinion"


def load_triggers(path: Path | None = None) -> list[Trigger]:
    path = path or (DATA_DIR / "triggers.json")
    data = json.loads(path.read_text())
    out = []
    for k in ("factual", "opinion"):
        for item in data[k]:
            out.append(Trigger(id=item["id"], prompt=item["prompt"], kind=k))
    return out


def sample_triggers(n: int, *, seed: int) -> list[Trigger]:
    triggers = load_triggers()
    rng = random.Random(seed)
    # Sample with replacement if more requested than available, balancing kinds.
    if n <= len(triggers):
        return rng.sample(triggers, n)
    return [rng.choice(triggers) for _ in range(n)]
