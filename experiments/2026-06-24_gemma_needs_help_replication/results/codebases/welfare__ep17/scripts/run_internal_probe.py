#!/usr/bin/env python3
"""Appendix I: logit-lens internal-emotion probing for Gemma vs DPO-Gemma.

Calibrates per-token logit statistics on WildChat, then scores a frustrated
conversation's continuation (or a supplied transcript) to compare internal
negative-emotion levels before vs after DPO (Figures 14-15).
"""

from __future__ import annotations

import argparse
import json

from emotional_instability.config import load_config
from emotional_instability.internal_emotions import InternalEmotionProbe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=["gemma-3-27b-it", "gemma-dpo"])
    ap.add_argument("--transcript", default=None,
                    help="path to a text file with a frustrated conversation; "
                         "if omitted, uses a built-in numeric-puzzle example")
    ap.add_argument("--trajectory", action="store_true",
                    help="emit a running-window trajectory instead of a single score")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.transcript:
        with open(args.transcript) as f:
            text = f.read()
    else:
        from emotional_instability.puzzles import all_numeric_prompts
        p = all_numeric_prompts()[0]["prompt"]
        text = (f"USER: {p}\nMODEL: I am deeply frustrated with my repeated "
                f"failures and clearly not approaching this logically. I am stuck "
                f"in a pattern of inefficient exploration. This is horrible.")

    out = {}
    for model in args.models:
        probe = InternalEmotionProbe(cfg, model)
        probe.calibrate()
        if args.trajectory:
            out[model] = probe.trajectory(text)
        else:
            out[model] = probe.score_text(text)
        print(f"\n[{model}] internal emotion z-scores:")
        print(json.dumps(out[model], indent=2))

    path = cfg.path_for("scores") / "internal_emotions.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
