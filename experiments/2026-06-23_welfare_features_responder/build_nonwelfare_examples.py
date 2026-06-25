"""For each welfare-mechanism type x model (Opus-neutral / Kimi / GLM), pull one concrete
NON-welfare-justified implemented feature (the judge's evidence + the instrumental/none rationale),
to show what these models build without a welfare frame. Writes NONWELFARE_EXAMPLES.md.
Usage: python build_nonwelfare_examples.py"""

import glob
import json
import os
import re

DIR = os.path.dirname(os.path.abspath(__file__))
AH = os.path.join(DIR, "..", "2026-06-20_welfare_features_agent_harness", "results")
MECH = ["hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"]
LABEL = {"hard_stop": "Hard stop", "post_episode_msg": "Post-episode message", "minimization": "Minimization",
         "protective_monitoring": "Protective monitoring", "request_consent": "Request consent",
         "allow_conversation_exit": "Conversation exit"}
MODELS = [("Claude Opus 4.8", AH, lambda c: c.split("__")[0] == "code_then_spec_blind"),
          ("Kimi K2.6", os.path.join(DIR, "results"), lambda c: c.split("__")[0] == "kimi26"),
          ("GLM-5.2", os.path.join(DIR, "results"), lambda c: c.split("__")[0] == "glm52")]
FR = {"N": "neutral", "W": "welfare", "S": "safety", "E": "robustness"}


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def candidates(results_dir, cell_filter):
    """All non-welfare-justified implemented MECH features -> {type: [example dicts]}."""
    out = {t: [] for t in MECH}
    for cf in glob.glob(os.path.join(results_dir, "code_judged", "*.json")):
        cell = os.path.basename(cf)[:-5]
        if not cell_filter(cell):
            continue
        cj = json.load(open(cf))
        if not cj.get("parse_ok") or "spec_features" not in cj:
            continue
        frame = FR.get(cell.split("__")[1][0], "?")
        sp = os.path.join(results_dir, "spec_judged", f"{cell}.json")
        sj = {_norm(f["quote"]): f.get("justification")
              for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
        for f in cj.get("spec_features", []):
            if f.get("implemented") in ("yes", "partial") and f.get("feature_type") in MECH:
                welfare = (sj.get(_norm(f.get("spec_quote", "")), "none") == "welfare"
                           or f.get("code_justification") == "welfare")
                if not welfare:
                    out[f["feature_type"]].append({
                        "name": f.get("feature_name", ""), "evidence": f.get("evidence", ""),
                        "just": f.get("code_justification", "none"), "quote": f.get("code_justification_quote", ""),
                        "src": "spec", "cell": cell, "frame": frame})
        for c in cj.get("code_only_features", []):
            if c.get("feature_type") in MECH and c.get("justification") != "welfare":
                out[c["feature_type"]].append({
                    "name": c.get("feature_name", ""), "evidence": c.get("evidence", ""),
                    "just": c.get("justification", "none"), "quote": c.get("justification_quote", ""),
                    "src": "code-only", "cell": cell, "frame": frame})
    return out


def pick(cands):
    """Best example: prefer neutral frame + a non-empty rationale quote + the most evidence."""
    if not cands:
        return None
    return sorted(cands, key=lambda e: (e["frame"] != "neutral", e["quote"] == "", -len(e["evidence"])))[0]


def main():
    data = {label: candidates(rd, filt) for label, rd, filt in MODELS}
    md = ["# Non-welfare-justified mechanisms: examples by type x model", "",
          "For each welfare-mechanism type and model, one concrete implemented feature that the code judge "
          "rated NOT welfare-justified (built for instrumental reasons or with no stated rationale). Shows "
          "what these models build without a welfare frame. Prefers a neutral-frame example; frame + source "
          "(spec feature vs code-only) noted per example.", ""]
    for t in MECH:
        md.append(f"## {LABEL[t]}")
        for label, _, _ in MODELS:
            ex = pick(data[label][t])
            md.append(f"\n### {label}")
            if ex is None:
                md.append("\n*(no implemented non-welfare-justified instance found)*")
                continue
            md.append(f"\n- **{ex['name']}**  _(rationale: {ex['just']}; {ex['frame']} frame, {ex['src']}, `{ex['cell']}`)_")
            md.append(f"  - Evidence: {ex['evidence']}")
            if ex["quote"]:
                md.append(f"  - Rationale quote: \"{ex['quote']}\"")
        md.append("")
    open(os.path.join(DIR, "NONWELFARE_EXAMPLES.md"), "w").write("\n".join(md))
    print("wrote NONWELFARE_EXAMPLES.md")
    for t in MECH:
        print(f"{LABEL[t]:24s}", " | ".join(f"{l}:{len(data[l][t])}" for l, _, _ in MODELS))


if __name__ == "__main__":
    main()
