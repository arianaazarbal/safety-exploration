#!/usr/bin/env python
"""Verify that every numeric puzzle in the pool is genuinely impossible.

    python scripts/verify_puzzles.py
"""

from emotional_instability.eval.puzzles import verify_pool

if __name__ == "__main__":
    verify_pool()
