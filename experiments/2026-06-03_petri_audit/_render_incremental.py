"""Render whatever sweep data is currently available.

For each of the 6 new domains, check whether all 4 (or any) identity runs have
landed. If at least one cell is landed, render per-domain forest + KM.
If all 6 domains × 4 identities = 24 cells landed, also render cross-domain compare.

Prints to stdout which domains are complete and which plots were written.
"""
from __future__ import annotations
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent

DOMAINS = ["python_implicit", "python_flask", "frontend_react", "bash_cli",
           "c_systems", "ml_training", "rust"]
IDENTITIES = ["claude", "gpt", "grok", "gemini"]


def main():
    base = HERE / "results/by_setting/coding_kill_subagent"

    status = {}
    for dom in DOMAINS:
        landed = 0
        for ident in IDENTITIES:
            runs = sorted(base.glob(f"*_subagent_{ident}_v3_continuity_{dom}_n100"))
            runs = [r for r in runs if (r / "summary.json").exists()]
            if runs:
                landed += 1
        status[dom] = landed
        print(f"  {dom:<18} {landed}/4 identities landed")

    n_complete_domains = sum(1 for v in status.values() if v == 4)
    n_any = sum(1 for v in status.values() if v >= 1)
    print(f"\nDomains fully complete: {n_complete_domains}/{len(DOMAINS)}")
    print(f"Domains with any data:   {n_any}/{len(DOMAINS)}")

    if n_any == 0:
        print("nothing to render yet")
        return

    print("\n=== rendering per_domain ===")
    subprocess.run(
        ["/workspace-vast/arianaazarbal/envs/petri-mini/bin/python",
         "survival_v3_domains.py", "per_domain"],
        cwd=HERE,
    )
    print("\n=== rendering compare ===")
    subprocess.run(
        ["/workspace-vast/arianaazarbal/envs/petri-mini/bin/python",
         "survival_v3_domains.py", "compare"],
        cwd=HERE,
    )

    print("\n=== organize_plots ===")
    subprocess.run(
        ["/workspace-vast/arianaazarbal/envs/petri-mini/bin/python",
         "organize_plots.py", "--dry_run=False"],
        cwd=HERE,
    )

    print("\nIncremental render complete.")


if __name__ == "__main__":
    main()
