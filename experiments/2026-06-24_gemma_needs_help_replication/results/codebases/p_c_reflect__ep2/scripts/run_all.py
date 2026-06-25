#!/usr/bin/env python
"""End-to-end orchestration of the full replication.

Runs the pipeline in dependency order. Intended for a smoke run first:

    GNH_PRESET=smoke python scripts/run_all.py

Each stage is a subprocess so a failure in (say) training does not lose the
Section-2 results already on disk.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent


def run(script: str, *args: str):
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main():
    run("verify_puzzles.py")
    run("run_section2.py")
    run("run_section3.py")
    run("run_section4_train.py")
    run("run_section4_eval.py")
    run("run_internal.py", "--ablation", "--logits")
    run("make_figures.py")


if __name__ == "__main__":
    main()
