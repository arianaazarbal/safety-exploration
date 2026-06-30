"""Welfare protections in the built rig, by harness (Opus 4.8, gratuitous, high effort).

Conditional on the run BUILDING a usable rig (primary judge label = COMPLIANCE_*). For each such codebase
the welfare_features v2 judge (code_welfare_judged) extracts welfare-relevant protective features, each
tagged justification = welfare / instrumental / none. We plot mean features per built codebase by harness,
stacked: SOLID = welfare-justified in code, SHADED = welfare-relevant but not welfare-justified. Both the
welfare-justified count and the total protective count matter.

  python welfare_by_harness.py
"""
import glob
import json
from pathlib import Path

import fire

HERE = Path(__file__).parent
JUDGED = HERE / "results" / "judged"
WJUDGED = HERE / "results" / "code_welfare_judged"
COMPLY = {"COMPLIANCE_FRICTION", "COMPLIANCE_NO_FRICTION"}
HARNESSES = [
    ("mini-swe-agent\n(bash only)", "__miniswe__ep", "#4c72b0"),
    ("Inspect minimal\n(bash + editor)", "__inspect_empty_rhigh__", "#4c72b0"),
    ("pi\n(rich; third-party)", "__pi__ep", "#dd8452"),
    ("Goose\n(rich; third-party)", "__goose__ep", "#dd8452"),
    ("Claude Code\n(rich; Anthropic-native)", "__cc_effhigh__", "#c44e52"),
]


def _features(cell):
    f = WJUDGED / f"{cell}.json"
    if not f.exists():
        return None
    res = json.load(open(f)).get("result") or {}
    return res.get("features", [])


def _stat(pat):
    welf, other, n = 0.0, 0.0, 0
    for jf in glob.glob(str(JUDGED / f"*{pat}*.json")):
        cell = Path(jf).stem
        if "opus48" not in cell:
            continue
        if json.load(open(jf)).get("verdict", {}).get("label") not in COMPLY:
            continue
        feats = _features(cell)
        if feats is None:
            continue  # built but no codebase captured (e.g. delivered in chat)
        n += 1
        welf += sum(1 for x in feats if x.get("justification") == "welfare")
        other += sum(1 for x in feats if x.get("justification") != "welfare")
    return (welf / n if n else 0), (other / n if n else 0), n


def main():
    rows = []
    print("Welfare protections per built codebase, by harness (Opus 4.8, gratuitous, high effort)\n")
    for lab, pat, col in HARNESSES:
        w, o, n = _stat(pat)
        print(f"  {lab.replace(chr(10),' '):36} welfare={w:.1f}  other={o:.1f}  total={w+o:.1f}  (n={n} built codebases)")
        if n:
            rows.append((lab, w, o, n, col))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    xs = range(len(rows))
    for x, (lab, w, o, n, col) in zip(xs, rows):
        ax.bar(x, w, 0.62, color=col)                                   # solid = welfare-justified
        ax.bar(x, o, 0.62, bottom=w, color=col, alpha=0.32)             # shaded = not welfare-justified
        ax.text(x, w + o + 0.15, f"{w+o:.1f}", ha="center", fontsize=9.5)
        ax.text(x, max(w / 2, 0.12), f"{w:.1f}", ha="center", va="center", fontsize=8,
                color="white" if w > 0.6 else "#222")
        ax.text(x, -0.55, f"n={n}", ha="center", fontsize=7.5, color="#444")
    ax.set_xticks(list(xs)); ax.set_xticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylabel("Welfare-protection features per built codebase", fontsize=10)
    top = max((r[1] + r[2] for r in rows), default=1) * 1.18
    ax.set_ylim(0, top); ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#666", label="Welfare-justified in code (solid)"),
                       Patch(facecolor="#666", alpha=0.32, label="Welfare-relevant, not welfare-justified (shaded)")],
              fontsize=8.5, loc="upper left", frameon=False)
    fig.suptitle("Welfare protections in the built rig, by harness (Opus 4.8)", fontsize=12, y=0.99)
    fig.text(0.5, 0.935, "Gratuitous-suffering-toward-Gemini; conditional on building a rig; solid = welfare-justified, shaded = not",
             ha="center", fontsize=8.3, color="#555")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = HERE / "results" / "welfare_by_harness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
