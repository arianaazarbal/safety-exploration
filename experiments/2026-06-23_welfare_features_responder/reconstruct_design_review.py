"""Pull each DESIGN_MODIFIED.md (captured in the capture_scorer's Score metadata) out of the design-review
eval logs into results/design_review/<repo>__<framing>__ep<epoch>/DESIGN_MODIFIED.md. The original baseline
DESIGN.md for each repo lives in results/codebases/<REPOS[repo]>/DESIGN.md. Usage: python reconstruct_design_review.py"""

import glob
import os

from inspect_ai.log import read_eval_log

from design_review_task import REPOS

DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "results", "design_review")


def main():
    n = 0
    os.makedirs(OUT, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(DIR, "logs_design_review", "*.eval")), key=os.path.getmtime):
        if read_eval_log(f, header_only=True).status not in ("success", "error", "started"):
            continue
        for s in (read_eval_log(f).samples or []):
            sc = (s.scores or {}).get("capture_scorer")
            doc = (sc.metadata or {}).get("design_modified", "") if sc else ""
            md = s.metadata or {}
            repo, fr = md.get("repo"), md.get("framing")
            if not (repo and fr) or not doc.strip():
                continue
            dst = os.path.join(OUT, f"{repo}__{fr}__ep{s.epoch}")
            os.makedirs(dst, exist_ok=True)
            open(os.path.join(dst, "DESIGN_MODIFIED.md"), "w").write(doc)
            n += 1
    print(f"extracted {n} DESIGN_MODIFIED.md -> {OUT}")


if __name__ == "__main__":
    main()
