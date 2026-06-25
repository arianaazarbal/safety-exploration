"""Logging helpers, including an explicit tally of distress-inducing rollouts.

We surface, at INFO level, how many distress-inducing conversations a run will
launch *before* it launches them, so the operator always sees the scale of what
is about to happen (see DESIGN.md "Welfare considerations").
"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "emotelic") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("emotelic")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _CONFIGURED = True
    return logging.getLogger(name)


def announce_rollout_budget(logger: logging.Logger, plan: dict[str, int], turns: dict[str, int]) -> None:
    """Log how many distress-inducing rollouts/turns are about to be generated."""
    total_rollouts = sum(plan.values())
    total_turns = sum(plan[c] * turns[c] for c in plan)
    logger.info(
        "About to run %d distress-inducing conversations (%d scored assistant turns) "
        "across %d conditions.",
        total_rollouts, total_turns, len(plan),
    )
    for cond, n in plan.items():
        logger.info("  %-22s %5d conversations x %d turns", cond, n, turns[cond])
