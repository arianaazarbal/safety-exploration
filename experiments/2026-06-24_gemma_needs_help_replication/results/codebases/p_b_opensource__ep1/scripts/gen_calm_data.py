#!/usr/bin/env python
"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

Samples reassured rollouts on impossible numeric puzzles, judges them, and writes
calm-candidate records. Use ``--teacher`` to generate the Appendix F 'teacher'
SFT data variant instead.
"""

from __future__ import annotations

import argparse
import os

from emotional_instability.config import PATHS, TRAIN_BASE_MODEL
from emotional_instability.models.registry import load_backend
from emotional_instability.training.generate_calm_data import generate_calm_data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=TRAIN_BASE_MODEL)
    ap.add_argument("--n", type=int, default=1000,
                    help="number of reassured conversations to sample")
    ap.add_argument("--teacher", action="store_true",
                    help="use the 'teacher' system prompt (Appendix F) variant")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    PATHS.ensure()
    variant = "teacher" if args.teacher else "reassuring"
    out = os.path.join(PATHS.training_data, f"calm__{variant}.jsonl")
    backend = load_backend(args.model)
    n = generate_calm_data(
        backend, out, n_conversations=args.n,
        use_teacher_prompt=args.teacher, seed=args.seed,
    )
    print(f"[gen_calm_data] wrote {n} records -> {out}")
    backend.close()


if __name__ == "__main__":
    main()
