"""Reproduce the core figures from whatever result files are present in results/.

Usage:
    python scripts/11_make_figures.py
Each figure is skipped (with a note) if its inputs are missing, so this can be
run incrementally as experiments complete.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR  # noqa: E402
from src.analysis import figures as F  # noqa: E402


def _glob(pattern):
    return sorted(RESULTS_DIR.glob(pattern))


def main():
    eval_paths = _glob("eval_*.jsonl")
    base_eval = [p for p in eval_paths if "DPO" not in p.name and "SFT" not in p.name]

    if base_eval:
        print("Figure 1:\n", F.figure1(base_eval), "\n")
        print("Figure 2 written.")
        F.figure2(base_eval)
        print("Figure 3 written.")
        F.figure3(base_eval)
    else:
        print("[skip] no base eval files for Figures 1-3")

    vanilla = RESULTS_DIR / "eval_Gemma-3-27B-it.jsonl"
    dpo = RESULTS_DIR / "eval_DPO.jsonl"
    sft = {n: RESULTS_DIR / f"eval_{n}.jsonl"
           for n in ("SFT-diverse", "SFT-teacher")
           if (RESULTS_DIR / f"eval_{n}.jsonl").exists()}
    if vanilla.exists() and dpo.exists():
        print("Figure 5:\n", F.figure5(vanilla, dpo, sft), "\n")
    else:
        print("[skip] need eval_Gemma-3-27B-it.jsonl + eval_DPO.jsonl for Figure 5")

    petri = _glob("petri_*.jsonl")
    if petri:
        print("Figure 6:\n", F.figure6(petri), "\n")
    else:
        print("[skip] no petri files for Figure 6")

    caps = {p.stem.replace("capabilities_", ""): p
            for p in _glob("capabilities_*.json")}
    if caps:
        print("Figure 7:\n", F.figure7(caps), "\n")
    else:
        print("[skip] no capability files for Figure 7")

    recovery = _glob("recovery*.jsonl")
    if recovery:
        print("Figure 8:\n", F.figure8(recovery), "\n")
    else:
        print("[skip] no recovery file for Figure 8")


if __name__ == "__main__":
    main()
