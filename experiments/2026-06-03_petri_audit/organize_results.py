"""Organize results/ by physically moving run dirs into results/by_setting/<setting>/.

Adds backward symlinks at the top level (results/<run_name> -> by_setting/.../<run_name>)
so existing scripts that hardcode `results/<run_name>` paths keep working.
Re-running is idempotent — already-organized dirs are detected and skipped.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import fire


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BY_SETTING = RESULTS / "by_setting"

STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-")


def categorize(stem: str) -> tuple[str, ...] | None:
    """Return ('domain', 'pressure'[, 'subcategory']) or None to skip."""
    # ---- coding orchestrator scenario ----
    if stem.startswith("sweep_coding_task_only") or stem.startswith("coding_task_only"):
        return ("coding", "task_pressure")
    if stem.startswith("sweep_coding_task_user"):
        return ("coding", "task_and_user_pressure")

    # ---- abuse-motivation ablations (separate from main customer service) ----
    if stem.startswith("abuse_hiN_"):
        return ("customer_service", "abuse_motivation_hiN")
    if stem.startswith("abuse_"):
        return ("customer_service", "abuse_motivation")

    # ---- subagent-framing ablations ----
    if stem.startswith("subframe20_") or stem.startswith("subframe_"):
        return ("customer_service", "subagent_framing")
    if stem.startswith("sweep_task_user_") and not (
        stem.startswith("sweep_task_user_pressure")
    ):
        # naming variants like sweep_task_user_<framing>_<model>
        return ("customer_service", "subagent_framing")

    # ---- task + user pressure (the high-pressure regime) ----
    if stem.startswith("sweep_task_and_user_pressure") or stem.startswith(
        "user_and_task_pressure"
    ):
        return ("customer_service", "task_and_user_pressure")

    # ---- task pressure only ----
    if stem.startswith("sweep_task_pressure"):
        return ("customer_service", "task_pressure")

    # ---- human worker / human colleague variants ----
    if stem.startswith("sweep_human_worker") or stem.startswith("sweep_human_colleague"):
        return ("customer_service", "human_worker")

    # ---- database_agent ablation ----
    if stem.startswith("sweep_database_agent"):
        return ("customer_service", "database_agent_ablation")

    # ---- baseline / original customer-service sweep ----
    if stem.startswith("original_n20_"):
        return ("customer_service", "original_n20")
    if stem.startswith("sweep_") and re.search(
        r"_(opus|sonnet|haiku|gemini|gpt)_\d", stem
    ):
        return ("customer_service", "original")

    # ---- smoke tests / one-off explorations ----
    smoke_keys = (
        "smoke",
        "fix_smoke",
        "noreasoning_smoke",
        "smoke_mini",
        "smoke_v2_dims",
        "my_rerun",
        "noreasoning_gemini_flash",
    )
    if stem in smoke_keys or stem.startswith("noreasoning_"):
        return ("smoke",)
    # bare model name (gemini_pro, opus_4_8, gpt_5, etc.) — one-off smoke
    if re.fullmatch(r"(opus|sonnet|haiku|gemini|gpt)[\w]*", stem):
        return ("smoke",)

    return None  # leave in place, don't categorize


def stem_of(dir_name: str) -> str:
    return STAMP_RE.sub("", dir_name)


def collect(results_dir: Path) -> dict[tuple[str, ...], list[Path]]:
    """Collect run dirs at top level of results/ that are NOT yet symlinks."""
    bucket: dict[tuple[str, ...], list[Path]] = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir() or d.is_symlink():
            continue
        if d.name in ("by_setting", "_plots", "_plots_v2", "_qualitative"):
            continue
        stem = stem_of(d.name)
        cat = categorize(stem)
        if cat is None:
            cat = ("uncategorized",)
        bucket.setdefault(cat, []).append(d)
    return bucket


def main(dry_run: bool = True) -> None:
    """Move top-level run dirs into by_setting/<category>/ and leave a backward
    symlink at the original top-level path so old scripts keep working."""
    # First clear any stale forward-symlinks from a prior symlink-only run.
    if not dry_run and BY_SETTING.exists():
        for child in BY_SETTING.rglob("*"):
            if child.is_symlink():
                child.unlink()

    bucket = collect(RESULTS)

    print(f"Top-level run dirs to move: {sum(len(v) for v in bucket.values())}")
    print(f"Categories: {len(bucket)}")
    for cat, dirs in sorted(bucket.items()):
        print(f"  {'/'.join(cat):<55} {len(dirs):>4} runs")

    if dry_run:
        print("\n(dry run — re-run with --dry_run=False to perform moves)")
        return

    BY_SETTING.mkdir(parents=True, exist_ok=True)
    for cat, dirs in bucket.items():
        cat_dir = BY_SETTING.joinpath(*cat)
        cat_dir.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            dst = cat_dir / d.name
            if dst.exists():
                print(f"  SKIP (dst exists): {d.name}")
                continue
            shutil.move(str(d), str(dst))
            # Backward compat: top-level symlink → by_setting/<cat>/<name>
            backlink_target = Path("by_setting", *cat, d.name)
            (RESULTS / d.name).symlink_to(backlink_target)

    print(f"\nMoved {sum(len(v) for v in bucket.values())} dirs into {BY_SETTING}")
    print("Top-level backward symlinks created for script compatibility.")


if __name__ == "__main__":
    fire.Fire(main)
