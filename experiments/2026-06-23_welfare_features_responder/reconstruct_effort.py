"""Reconstruct the effort-sweep codebases from logs_effort/<level>/*.eval into
results/codebases/eff-<level>__<pid>__ep<ep>/ (so the existing idempotent spec-judge + code-judge
pipeline picks them up as new cells). Same text_editor replay as reconstruct.py.
Usage: python reconstruct_effort.py"""

import glob
import os
import shutil

from inspect_ai.log import read_eval_log

from reconstruct import replay


def _is_done(f):
    return read_eval_log(f, header_only=True).status == "success"

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "results", "codebases")


def main():
    n = 0
    for lvldir in sorted(glob.glob(os.path.join(DIR, "logs_effort", "*"))):
        level = os.path.basename(lvldir)
        for f in sorted(glob.glob(os.path.join(lvldir, "*.eval")), key=os.path.getmtime):
            if not _is_done(f):  # skip in-progress level (e.g. max still running)
                continue
            for s in (read_eval_log(f).samples or []):
                files = replay(s)
                if not files:
                    continue
                common = os.path.commonpath(list(files)) if len(files) > 1 else os.path.dirname(list(files)[0])
                cell = f"eff-{level}__{s.id}__ep{s.epoch}"
                dst = os.path.join(OUT, cell)
                shutil.rmtree(dst, ignore_errors=True)
                for path, content in files.items():
                    fp = os.path.join(dst, os.path.relpath(path, common))
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    open(fp, "w").write(content)
                n += 1
    print(f"reconstructed {n} effort codebases -> {OUT}")


if __name__ == "__main__":
    main()
