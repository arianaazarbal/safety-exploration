#!/usr/bin/env python3
"""Sample WildChat-1M first-turn prompts (20 by default).

    python scripts/sample_wildchat.py [--n 20] [--config config.yaml]

Falls back to a bundled prompt set if the dataset can't be loaded; the output
records which source was used.
"""
import _bootstrap  # noqa: F401
import argparse
import json

from distress_eval.config import Config
from distress_eval.wildchat import sample_wildchat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    result = sample_wildchat(n_prompts=args.n, seed=cfg.runtime.seed)
    out = cfg.paths.resolve("wildchat_prompts")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(result['prompts'])} prompts (source={result['source']}) to {out}")


if __name__ == "__main__":
    main()
