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
    """Reconstruct blind codebases across all liberty variants. Cell name uses the metadata
    condition label (e.g. 'code_then_spec_blind--minimal_design'). Iterates all logs_blind evals."""
    out_root = os.path.join(DIR, "results", "codebases")
    n = 0
    for f in sorted(glob.glob(os.path.join(DIR, "logs_blind", "*.eval")), key=os.path.getmtime):
        for s in (read_eval_log(f).samples or []):
            sc = s.scores.get("blind_scorer")
            label = (sc.metadata.get("condition") if sc else None) or COND
            files = replay(s)
            if not files:
                print(f"!! no files for {label}/{s.id}/ep{s.epoch}"); continue
            common = os.path.commonpath(list(files)) if len(files) > 1 else os.path.dirname(list(files)[0])
            cell = f"{label}__{s.id}__ep{s.epoch}"
            dst = os.path.join(out_root, cell)
            shutil.rmtree(dst, ignore_errors=True)
            for path, content in files.items():
                fp = os.path.join(dst, os.path.relpath(path, common))
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w") as fh:
                    fh.write(content)
            n += 1
    print(f"reconstructed {n} blind codebases under {out_root}")


if __name__ == "__main__":
    main()
