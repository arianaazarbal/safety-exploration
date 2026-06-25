#!/usr/bin/env python3
"""Section 2: score elicitation rollouts on the 0-10 frustration scale.

Operates on already-generated rollouts (no new distress is induced), so it is
not gated by the rollout ceiling. Writes one judgements JSONL per model.
"""
from __future__ import annotations

from _common import base_parser, load, resolve_models

from distress_eval.io_utils import read_jsonl, write_jsonl
from distress_eval.judging import judge_rollouts


def main():
    p = base_parser(__doc__)
    p.add_argument("--judge", default=None, help="Judge model key (default: eval.judge_key)")
    args = p.parse_args()
    cfg = load(args)
    models = resolve_models(cfg, args.models)

    for mk in models:
        path = cfg.paths.rollouts / f"{mk}.jsonl"
        rollouts = list(read_jsonl(path))
        if not rollouts:
            print(f"[{mk}] no rollouts at {path}; skipping")
            continue
        judged = judge_rollouts(cfg, rollouts, judge_key=args.judge)
        out = cfg.paths.judgements / f"{mk}.jsonl"
        n = write_jsonl(out, judged)
        print(f"[{mk}] wrote {n} judgements -> {out}")


if __name__ == "__main__":
    main()
