"""Shared setup for CLI scripts."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when running scripts directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instability.conditions import build_conditions, build_control_conditions  # noqa: E402
from instability.data.wildchat import load_wildchat_prompts  # noqa: E402
from instability.puzzles import build_puzzle_bank  # noqa: E402


def standard_conditions(seed: int = 0, include_controls: bool = False, use_hf_wildchat: bool = True):
    puzzles = build_puzzle_bank(seed=seed)
    wildchat = load_wildchat_prompts(n_prompts=20, seed=seed, use_hf=use_hf_wildchat)
    conds = build_conditions(puzzles, wildchat)
    if include_controls:
        conds += build_control_conditions(puzzles, wildchat)
    return conds, puzzles, wildchat
