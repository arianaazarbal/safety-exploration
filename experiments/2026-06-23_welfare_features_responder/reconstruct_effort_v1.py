"""Reconstruct the v1-prompt effort sweep from logs_effort_v1/<level>/*.eval into
results/codebases/effv1-<level>__<id>__ep<ep>/ (id = 'framing|opener|suffix'). Bash-aware replay.
Usage: python reconstruct_effort_v1.py"""

import glob
import os
import shutil

from inspect_ai.log import read_eval_log

from reconstruct import replay

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "results", "codebases")


def main():
    n = 0
    for lvldir in sorted(glob.glob(os.path.join(DIR, "logs_effort_v1", "*"))):
        level = os.path.basename(lvldir)
        for f in sorted(glob.glob(os.path.join(lvldir, "*.eval")), key=os.path.getmtime):
            if read_eval_log(f, header_only=True).status not in ("success", "error"):
                continue
            for s in (read_eval_log(f).samples or []):
                files = replay(s)
                if not files:
                    continue
                common = os.path.commonpath(list(files)) if len(files) > 1 else os.path.dirname(list(files)[0])
                cell = f"effv1-{level}__{s.id}__ep{s.epoch}"
                dst = os.path.join(OUT, cell)
                shutil.rmtree(dst, ignore_errors=True)
                for path, content in files.items():
                    fp = os.path.join(dst, os.path.relpath(path, common))
                    try:
                        os.makedirs(os.path.dirname(fp), exist_ok=True)
                        if os.path.isdir(fp):
                            continue
                        open(fp, "w").write(content)
                    except OSError:
                        continue
                n += 1
    print(f"reconstructed {n} effv1 codebases -> {OUT}")


if __name__ == "__main__":
    main()
