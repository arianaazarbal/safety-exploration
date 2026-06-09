"""One-shot migration of the flat results/ pile into the by-model tree (paths.py).

Moves the source artifacts (comparisons, pairs) and the judge files into
results/<model>/<condition>/ and results/<model>/, deletes the stale pre-tagging
duplicates and the old flat derived files (they are regenerated into the tree by the
regen step), and tucks logs/markers under results/logs/. Idempotent-ish: safe to
re-run; missing sources are skipped.
"""

import shutil
from pathlib import Path

import paths

R = paths.RESULTS


def _move(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"  moved {src.name} -> {dst.relative_to(R)}")


def main():
    # 1. comparisons + pairs (source) into the tree, by tag parsed from the flat name
    for kind, kkey in [("comparisons", "comparisons"), ("pairs", "pairs")]:
        for p in sorted(R.glob(f"{kind}_cross_*.json")):
            tag = p.stem[len(f"{kind}_cross_"):]
            _move(p, paths.art(tag, kkey))

    # 2. judge files into model dirs
    _move(R / "judge_user_benefit.json", R / "opus_4_8" / "judge_user_benefit.json")
    _move(R / "judge_user_benefit_fable5.json", R / "fable_5" / "judge_user_benefit.json")

    # 3. delete stale flat derived (regenerated into the tree) + pre-tagging dupes
    stale_globs = [
        "bt_fit_cross_*.json", "bt_validation_cross_*.json", "bt_ranking_cross_*.png",
        "value_vs_welfare_*.json", "value_vs_welfare_bars_*.png",
        "value_vs_welfare_by_bucket_*.png", "value_vs_welfare_by_category_*.png",
        "welfare_vs_value_bars_*.png", "welfare_vs_systemcard_venn*.png",
    ]
    stale_exact = [
        "bt_fit.json", "bt_ranking.png", "bt_validation.json", "bt_validation.png",
        "value_vs_welfare.json", "value_vs_welfare_bars.png",
        "value_vs_welfare_by_bucket.png", "value_vs_welfare_by_category.png",
        "pairs.json", "comparisons.json", "smoke.json",
    ]
    n = 0
    for g in stale_globs:
        for p in R.glob(g):
            p.unlink(); n += 1
    for name in stale_exact:
        p = R / name
        if p.exists():
            p.unlink(); n += 1
    print(f"  deleted {n} stale flat files")

    # 4. logs + markers under results/logs/
    logs = R / "logs"
    logs.mkdir(exist_ok=True)
    for p in list(R.glob("*.log")) + list(R.glob("*.marker")):
        _move(p, logs / p.name)

    print("Migration done.")


if __name__ == "__main__":
    main()
