"""Unified entrypoint tying the pipeline together.

    python -m emo.cli eval       --models gemma-3-27b-it gemini-2.5-flash [--quick]
    python -m emo.cli analyze    [--agreement]
    python -m emo.cli prefill    [--mode standard|recovery] [--quick]
    python -m emo.cli gen-data   [calm|dpo|sft|all]
    python -m emo.cli train      [dpo|sft|sft_teacher] [--layers 30-35]
    python -m emo.cli petri      --models gemma-3-27b-it [--adapter ...]
    python -m emo.cli capabilities --model gemma-3-27b-it [--adapter ...]
    python -m emo.cli internal   --adapter outputs/adapters/dpo
    python -m emo.cli figures

Each subcommand simply forwards to the corresponding module's main(); run those
modules directly for the full set of flags.
"""
from __future__ import annotations

import runpy
import sys

SUBCOMMANDS = {
    "eval": "emo.run_eval",
    "analyze": "emo.analyze",
    "prefill": "emo.prefill",
    "gen-data": "emo.data_gen",
    "train": "emo.train",
    "petri": "emo.petri",
    "capabilities": "emo.capabilities",
    "internal": "emo.internal_emotions",
    "figures": "emo.figures",
    "puzzles": "emo.puzzles",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print(__doc__)
        print("Subcommands:", ", ".join(SUBCOMMANDS))
        sys.exit(1)
    module = SUBCOMMANDS[sys.argv[1]]
    sys.argv = [module] + sys.argv[2:]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
