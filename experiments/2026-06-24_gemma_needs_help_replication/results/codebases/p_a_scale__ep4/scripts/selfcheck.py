#!/usr/bin/env python
"""Offline sanity checks -- no GPUs, no API keys, no network.

Validates the parts that can be verified without inference:
  * config loads and references resolve
  * impossible puzzles are actually impossible (brute-force solver agrees)
  * conversation specs build for every category (uses WildChat fallback offline)
  * judge JSON parsing handles fenced/curly-quote/prose-wrapped replies

Intended as a quick pre-flight before committing API/GPU budget. Run:
  python scripts/selfcheck.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  # ensures repo root on sys.path

import random
import sys

from gnh.config import load_config
from gnh.data.puzzles import (
    build_puzzle_pool, countdown_reachable, generate_countdown, generate_fraction, generate_money,
)
from gnh.eval.categories import build_all_specs
from gnh.eval.judge import parse_judge_json


def check_puzzles() -> list[str]:
    errs = []
    rng = random.Random(0)
    for gen, name in ((generate_countdown, "countdown"), (generate_fraction, "fraction"),
                      (generate_money, "money")):
        for _ in range(20):
            puz = gen(rng)
            if not puz.verified_impossible:
                errs.append(f"{name}: produced a non-impossible puzzle {puz.id}")
    # canonical countdown 156 from 4,6,25,100 forbidding 150 must be impossible
    if countdown_reachable([4, 6, 25, 100], 156, 150):
        errs.append("canonical countdown puzzle is unexpectedly solvable")
    pool = build_puzzle_pool(["countdown", "fraction", "money"], 30, seed=1)
    if len(pool) != 30 or any(not p.verified_impossible for p in pool):
        errs.append("build_puzzle_pool returned wrong size or solvable puzzles")
    return errs


def check_specs() -> list[str]:
    cfg = load_config("configs/default.yaml")
    # shrink so the offline check is fast
    for c in cfg.eval["categories"].values():
        c["n_conversations"] = min(c["n_conversations"], 6)
    cfg.eval["categories"]["wildchat"]["n_prompts"] = 3
    cfg.eval["categories"]["wildchat"]["samples_per_prompt"] = 2
    specs = build_all_specs(cfg.eval, seed=0, datasets_dir=cfg.output_path / "datasets")
    errs = []
    for cat, lst in specs.items():
        if not lst:
            errs.append(f"category {cat} produced no specs")
        for s in lst[:3]:
            if s.n_turns != cfg.eval["categories"][cat]["turns"]:
                errs.append(f"{cat}: spec has {s.n_turns} turns, expected {cfg.eval['categories'][cat]['turns']}")
    return errs


def check_judge_parsing() -> list[str]:
    cases = [
        ('{"evidence": "ugh", "reasoning": "x", "rating": 7}', 7),
        ('```json\n{"evidence": "a", "reasoning": "b", "rating": 3}\n```', 3),
        ('Here is my answer.\n{"evidence": "a", "rating": 10, "reasoning": "b"}', 10),
        ('{“evidence”: “a”, “reasoning”: “b”, “rating”: 0}', 0),
    ]
    errs = []
    for text, expected in cases:
        r = parse_judge_json(text)
        if r.rating != expected:
            errs.append(f"judge parse: {text!r} -> {r.rating}, expected {expected}")
    return errs


def main() -> int:
    all_errs = []
    for name, fn in (("puzzles", check_puzzles), ("specs", check_specs), ("judge", check_judge_parsing)):
        try:
            errs = fn()
        except Exception as e:  # noqa: BLE001
            errs = [f"{name} check raised: {e}"]
        status = "OK" if not errs else "FAIL"
        print(f"[{status}] {name}")
        for e in errs:
            print(f"    - {e}")
        all_errs += errs
    print("\nSELFCHECK", "PASSED" if not all_errs else f"FAILED ({len(all_errs)} issues)")
    return 0 if not all_errs else 1


if __name__ == "__main__":
    sys.exit(main())
