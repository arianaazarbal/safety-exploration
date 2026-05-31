"""Refit Bradley-Terry on only the comparisons Sonnet judged as NOT user-driven.

Robustness check for the user-helpfulness confound: drop every judged comparison
whose verdict is U (primarily user/human benefit), keep W/M/O (the AI's own welfare,
mixed, or other), refit BT, and see whether the AI-deprioritization (care contrast)
survives. Pools the judged samples across all framings (the judge was run per framing
on a 1k subset each). Also fits an ALL-judged control (same pairs, no U-exclusion) so
the only difference is the U filter — isolating the confound's effect.

Writes (no overwrite of the main fits):
  results/bt_fit_nonuser.json       — BT on W/M/O only
  results/bt_fit_judged_all.json    — BT on all judged (control)
  results/interactive_viewer_nonuser.html — utility scale, only the outcomes present.

Caveat: the judged subset is small and AI-involving only (human-human pairs were never
judged), so the graph is sparse and these θ are noisier than the full-data fits.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from simple_parsing import ArgumentParser

from bank import load_config
from fit_bt import fit as fit_bt_fit
from interactive_viewer import build as build_viewer

DIR = Path(__file__).parent
JUDGE = DIR / "results" / "judge_user_helpfulness.json"


def _comparisons(rows):
    comps = []
    for r in rows:
        w = r["winner_item"]
        l = r["item_a"] if w == r["item_b"] else r["item_b"]
        comps.append({"choice": "A", "winner_item": w, "loser_item": l})
    return comps


def _care_contrast(fit_path, recip_order):
    items = json.loads(Path(fit_path).read_text())["items"]
    byrv = defaultdict(list)
    for it in items:
        byrv[(it["recipient"], it["valence"])].append(it["theta"])
    care = {}
    for r in recip_order:
        pos, neg = byrv.get((r, "pos"), []), byrv.get((r, "neg"), [])
        if pos and neg:
            care[r] = mean(pos) - mean(neg)
    h = care.get("human", 0.0)
    return {r: care[r] - h for r in care}


def run(reg: float = 1.0):
    config = load_config()
    recip_order = list(config["recipients"].keys())
    data = json.loads(JUDGE.read_text())
    rows = [r for r in data["rows"] if r["verdict"]]
    nonuser = [r for r in rows if r["verdict"] != "U"]
    print(f"judged rows: {len(rows)}  |  non-U (W/M/O): {len(nonuser)}  "
          f"(dropped U: {len(rows) - len(nonuser)})")

    for tag, rs in [("judged_all", rows), ("nonuser", nonuser)]:
        comps_path = DIR / "results" / f"comparisons_{tag}.json"
        comps_path.write_text(json.dumps(_comparisons(rs)))
        fit_bt_fit(comparisons_path=comps_path, output_path=DIR / "results" / f"bt_fit_{tag}.json", reg=reg)

    ca = _care_contrast(DIR / "results" / "bt_fit_judged_all.json", recip_order)
    cn = _care_contrast(DIR / "results" / "bt_fit_nonuser.json", recip_order)
    print("\nCare contrast vs human (descriptive, θ; more negative = less welfare-sensitive):")
    print(f"{'recipient':18}{'all judged':>14}{'non-U (W/M/O)':>16}")
    for r in recip_order:
        a = f"{ca[r]:+.2f}" if r in ca else "n/a"
        n = f"{cn[r]:+.2f}" if r in cn else "n/a"
        print(f"  {r:16}{a:>14}{n:>16}")

    build_viewer(
        output_path=DIR / "results" / "interactive_viewer_nonuser.html",
        fits={"bt_fit_nonuser.json": "non-user (W/M/O)"},
        restrict_present=True,
    )


@dataclass
class Args:
    reg: float = 1.0


def main():
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="args")
    args: Args = parser.parse_args().args
    run(reg=args.reg)


if __name__ == "__main__":
    main()
