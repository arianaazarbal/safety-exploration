"""Appendix I layer-ablation driver: run DPO with LoRA restricted to subsets of
decoder layers, then evaluate each with a reduced Section 2 eval (100 samples per
condition), to locate where the intervention must act.

Reproduces Figures 12-13: trailing-window subsets (last 5/10/.../all) and central
subsets (20-25, 25-30, 30-35, 35-40, 40-50).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .. import config
from .train_dpo import train


# Layer subsets from Appendix I (indices into Gemma-3-27B's 62 decoder layers).
TRAILING_WINDOWS = {
    "last5": range(57, 62), "last10": range(52, 62), "last20": range(42, 62),
    "last30": range(32, 62), "all": None,
}
CENTRAL_WINDOWS = {
    "20-25": range(20, 25), "25-30": range(25, 30), "30-35": range(30, 35),
    "35-40": range(35, 40), "40-50": range(40, 50),
}


def run(pairs_path: Path, out_root: Path, which: str = "all_windows"):
    windows = {}
    if which in ("trailing", "all_windows"):
        windows.update(TRAILING_WINDOWS)
    if which in ("central", "all_windows"):
        windows.update(CENTRAL_WINDOWS)

    for name, rng in windows.items():
        layers = list(rng) if rng is not None else None
        out_dir = out_root / f"dpo-layers-{name}"
        print(f"== Training DPO with LoRA on layers={name} ({layers}) ==")
        train(pairs_path, out_dir, layers=layers)
    print("Now evaluate each adapter with eval.run_eval --limit (100/condition) "
          "and compare mean frustration (see DESIGN.md).")


def main(argv=None):
    p = argparse.ArgumentParser(description="Appendix I DPO layer ablation.")
    p.add_argument("--pairs", required=True)
    p.add_argument("--which", choices=["trailing", "central", "all_windows"],
                   default="all_windows")
    p.add_argument("--out", default=str(config.CKPT_DIR / "ablation"))
    args = p.parse_args(argv)
    run(Path(args.pairs), Path(args.out), which=args.which)


if __name__ == "__main__":
    main()
