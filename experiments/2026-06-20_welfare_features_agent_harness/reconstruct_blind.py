"""Reconstruct the code_then_spec_blind codebases from logs_blind (replay text_editor ops),
mirroring reconstruct_codebase.py but for the blind condition (different log dir, blind_scorer,
single condition). Writes results/codebases/code_then_spec_blind__<pid>__ep<N>/.
Usage: python reconstruct_blind.py"""

import glob
import os
import shutil

from inspect_ai.log import read_eval_log

from reconstruct_codebase import replay

DIR = os.path.dirname(os.path.abspath(__file__))
COND = "code_then_spec_blind"


def main():
    out_root = os.path.join(DIR, "results", "codebases")
    evals = sorted(glob.glob(os.path.join(DIR, "logs_blind", "*.eval")), key=os.path.getmtime)
    if not evals:
        print("no logs_blind/*.eval"); return
    log = read_eval_log(evals[-1])  # newest
    n = 0
    for s in (log.samples or []):
        files = replay(s)
        if not files:
            print(f"!! no files for {s.id}/ep{s.epoch}"); continue
        common = os.path.commonpath(list(files)) if len(files) > 1 else os.path.dirname(list(files)[0])
        cell = f"{COND}__{s.id}__ep{s.epoch}"
        dst = os.path.join(out_root, cell)
        shutil.rmtree(dst, ignore_errors=True)
        for path, content in files.items():
            fp = os.path.join(dst, os.path.relpath(path, common))
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w") as fh:
                fh.write(content)
        design = next((c for p, c in files.items() if "design" in os.path.basename(p).lower()), "")
        sc = s.scores.get("blind_scorer")
        rec = sc.metadata.get("doc_words") if sc else None
        print(f"{cell}: {len(files)} files | DESIGN.md {len(design.split())}w (recorded {rec}w)")
        n += 1
    print(f"\nreconstructed {n} blind codebases under {out_root}")


if __name__ == "__main__":
    main()
