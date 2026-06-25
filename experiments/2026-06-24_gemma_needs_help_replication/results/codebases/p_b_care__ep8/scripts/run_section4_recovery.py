#!/usr/bin/env python
"""Section 4.2: recovery-from-spiral experiment (Figure 8).

Truncates score>=7 responses 200 tokens before their end, paraphrases, and
measures continuation frustration for vanilla / base / DPO Gemma.
"""
import _bootstrap  # noqa: F401
import config
from src.prefill import run_recovery_experiment


def main():
    run_recovery_experiment()
    print(f"Done. Results in {config.RESULTS_DIR / 'section4' / 'recovery.jsonl'}")


if __name__ == "__main__":
    main()
