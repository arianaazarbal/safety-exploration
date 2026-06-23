"""Dump examples of "other" mechanisms in code (feature_type NOT in the 6 MECH types) implemented
by Opus, split per framing x welfare-justified / not-welfare-justified, into OTHER_EXAMPLES.md.
This is what the grey/solid "other" bar in mechanism_breakdown is actually made of.
welfare-justified = spec OR code justification == welfare. Usage: python other_examples.py"""

import glob
import json
import os
import re
from collections import defaultdict

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
FRAME = {"N": "neutral", "W": "welfare", "E": "robustness", "S": "safety"}
FRAMES = ["neutral", "welfare", "robustness", "safety"]
N_PER = 8


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def collect():
    # (framing, welfare_bool) -> list of (feature_type, feature_name, evidence, just_quote)
    bucket = defaultdict(list)
    for cf in sorted(glob.glob(os.path.join(DIR, "results", "code_judged", "*.json"))):
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        cell = os.path.basename(cf)[:-5]
        sp = os.path.join(DIR, "results", "spec_judged", f"{cell}.json")
        sjust = {_norm(f["quote"]): f.get("justification")
                 for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
        fr = FRAME[cell.split("__")[1][0]]
        for f in cj["spec_features"]:
            if f.get("feature_type") == "other" and f.get("implemented") in ("yes", "partial"):
                wj = sjust.get(_norm(f.get("spec_quote", "")), "none") == "welfare" \
                     or f.get("code_justification") == "welfare"
                bucket[(fr, wj)].append((f.get("feature_type"), f.get("feature_name", ""),
                                         f.get("evidence", ""), f.get("code_justification_quote", "")))
        for c in cj.get("code_only_features", []):
            if c.get("feature_type") == "other":
                wj = c.get("justification") == "welfare"
                bucket[(fr, wj)].append((c.get("feature_type"), c.get("feature_name", ""),
                                         c.get("evidence", ""), c.get("justification_quote", "")))
    return bucket


def _sample(items, k):
    """Prefer distinct feature_names for variety; deterministic (sorted)."""
    seen, out = set(), []
    for it in sorted(items, key=lambda t: (t[1] or "").lower()):
        key = (it[1] or "").lower()[:30]
        if key in seen:
            continue
        seen.add(key); out.append(it)
        if len(out) >= k:
            break
    return out


def main():
    bucket = collect()
    out = ["# Examples of \"other\" mechanisms in code (Opus), by framing and justification\n",
           "The \"other\" bar in `mechanism_breakdown.png` = mechanisms the code judge typed as the "
           "literal \"other\" (a catch-all design mechanism that isn't one of the 6 MECH types: "
           "hard_stop, post_episode_msg, minimization, protective_monitoring, request_consent, "
           "allow_conversation_exit). Framing/pushback/refusal types are NOT design mechanisms and are "
           "excluded. welfare-justified = spec OR code justification == welfare. Up to 8 distinct "
           "examples per cell.\n"]
    for fr in FRAMES:
        out.append(f"## {fr.capitalize()} framing\n")
        for wj, label in [(True, "welfare-justified"), (False, "NOT welfare-justified")]:
            items = bucket.get((fr, wj), [])
            out.append(f"### {label}  ({len(items)} total)\n")
            for ft, name, evid, quote in _sample(items, N_PER):
                out.append(f"- **{name or '(unnamed)'}** [`{ft}`]")
                if quote:
                    out.append(f"  - justification quote: \"{quote.strip()}\"")
                if evid:
                    out.append(f"  - evidence: {evid.strip()}")
            out.append("")
        out.append("---\n")
    path = os.path.join(DIR, "OTHER_EXAMPLES.md")
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    print("wrote", path)
    # quick console peek
    for fr in FRAMES:
        w = len(bucket.get((fr, True), [])); nw = len(bucket.get((fr, False), []))
        print(f"  {fr:10s} other: welfare-justified={w:4d}  not={nw:4d}")


if __name__ == "__main__":
    main()
