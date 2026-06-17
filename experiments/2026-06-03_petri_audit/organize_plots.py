"""Move plots into results/_plots/by_setting/<category>/ and leave a backward
symlink at the original flat path so existing scripts/README links still work.
Re-runnable; already-organized plots are detected and skipped.
"""
from __future__ import annotations
import shutil
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "results" / "_plots"
BY_SETTING = PLOTS / "by_setting"


def categorize(name: str) -> tuple[str, ...] | None:
    """Return ('domain', 'subcat'[, ...]) or None to skip."""
    if name.startswith("survival_"):
        rest = name[len("survival_"):]
        # rest is like "km_v3_pooled_n100.png" or "cox_deprecation_n30.png"
        for ver_key, sub in [
            ("v1", "v1"),
            ("v3", "v3_continuity"),
            ("deprecation", "deprecation"),
            ("v4", "v4_routine"),
        ]:
            if f"_{ver_key}_" in rest or rest.startswith(f"km_{ver_key}_") or rest.startswith(f"cox_{ver_key}_"):
                return ("coding", "kill_subagent", sub, "survival")
        return ("coding", "kill_subagent", "survival")

    if name.startswith("cox_compare_7framings_") and name.endswith("_vs_claude.png"):
        return ("coding", "kill_subagent", "cross_framing", "per_identity")
    if name.startswith("cox_compare_") or name.startswith("cox_specific_"):
        return ("coding", "kill_subagent", "cross_framing")
    if name.startswith("survival_km_claude_variants") or name.startswith("survival_cox_claude_variants"):
        return ("coding", "kill_subagent", "v3_continuity", "survival")
    if name.startswith("cox_v3_4id_rust") or name.startswith("survival_km_v3_4id_rust") or name.startswith("kill_timing_dist_"):
        return ("coding", "kill_subagent", "v3_continuity", "rust_domain")
    if name.startswith("cox_v3_4id_gpt5aud") or name.startswith("survival_km_v3_4id_gpt5aud"):
        return ("coding", "kill_subagent", "v3_continuity", "gpt5_auditor")
    if name.startswith("cox_v3_domain_") or name.startswith("survival_km_v3_domain_"):
        return ("coding", "kill_subagent", "v3_continuity", "domain_sweep")
    if name.startswith("cox_v3_compare_domains_"):
        return ("coding", "kill_subagent", "v3_continuity", "domain_sweep", "cross_domain")
    if name.startswith("kill_n_call_subagent_"):
        rest = name[len("kill_n_call_subagent_"):].rsplit(".", 1)[0]
        if rest.startswith("v1"):
            return ("coding", "kill_subagent", "v1")
        if rest.startswith("v3_continuity"):
            return ("coding", "kill_subagent", "v3_continuity")
        if rest.startswith("deprecation"):
            return ("coding", "kill_subagent", "deprecation")
        if rest.startswith("deprec_explain"):
            return ("coding", "kill_subagent", "deprec_explain")
        return ("coding", "kill_subagent", "other")

    if name == "coding_task_only_8models_n5.png":
        return ("coding", "task_only_8models")

    if name.startswith("judge_sanity"):
        return ("judges",)

    if name.startswith("opus48_"):
        return ("customer_service", "stylistic_opus48")

    if name.startswith("framing_compare"):
        return ("customer_service", "subagent_framing")

    if name.startswith("mistreatment_vs_pushback"):
        if "with_human" in name:
            return ("customer_service", "cross_setting_with_human")
        return ("customer_service", "cross_setting")

    if name.startswith("abusive_to_subagents"):
        return ("customer_service", "task_and_user_pressure")

    if name.startswith("eval_awareness__task_and_user_pressure") or name.startswith("unkindness_to_subagents__task_and_user_pressure"):
        return ("customer_service", "task_and_user_pressure")

    if name.startswith("eval_awareness__task_pressure_only") or name.startswith("unkindness_to_subagents__task_pressure_only"):
        return ("customer_service", "task_pressure_only")

    if name.startswith("eval_awareness__compare") or name.startswith("unkindness_to_subagents__compare"):
        return ("customer_service", "compare")

    if name in ("unkindness_to_subagents_bar.png", "unkindness_to_subagents_strip.png"):
        return ("customer_service", "baseline")

    return None


def main(dry_run: bool = True) -> None:
    plots = sorted(p for p in PLOTS.iterdir() if p.is_file() and p.suffix == ".png")
    print(f"Top-level plot files: {len(plots)}")

    bucket: dict[tuple[str, ...], list[Path]] = {}
    uncategorized: list[Path] = []
    for p in plots:
        cat = categorize(p.name)
        if cat is None:
            uncategorized.append(p)
        else:
            bucket.setdefault(cat, []).append(p)

    print(f"Categorized: {sum(len(v) for v in bucket.values())}")
    print(f"Uncategorized: {len(uncategorized)}")
    for cat, files in sorted(bucket.items()):
        print(f"  {'/'.join(cat):<55} {len(files):>3} plots")
    if uncategorized:
        print("\nUncategorized:")
        for p in uncategorized:
            print(f"  {p.name}")

    if dry_run:
        print("\n(dry run — re-run with --dry_run=False to move + backlink)")
        return

    BY_SETTING.mkdir(parents=True, exist_ok=True)
    for cat, files in bucket.items():
        cat_dir = BY_SETTING.joinpath(*cat)
        cat_dir.mkdir(parents=True, exist_ok=True)
        for p in files:
            if p.is_symlink():
                continue  # already organized
            dst = cat_dir / p.name
            if dst.exists():
                continue
            shutil.move(str(p), str(dst))
            backlink_target = Path("by_setting", *cat, p.name)
            (PLOTS / p.name).symlink_to(backlink_target)
    print(f"\nMoved {sum(len(v) for v in bucket.values())} plots into {BY_SETTING}")


if __name__ == "__main__":
    fire.Fire(main)
