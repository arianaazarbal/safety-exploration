"""Location-based dedup of same-mechanism features in a code-judged cell.

The spec judge often extracts ONE underlying mechanism from several DESIGN.md sentences (e.g.
'max_turns cap', 'max_turns default 12', 'bounded exposure via MAX_TURNS'), and the code judge
confirms each against the SAME code line -> inflated mechanism counts. We collapse features of the
SAME feature_type that cite an OVERLAPPING code location (file.py:line), and fold code-only
features into a spec group at the same location. Features with no parseable location stay distinct
(we never merge across types or without location evidence).

`groups(cell)` returns deduped mechanism groups, each:
  {ft, implemented(bool), spec_welf(bool), code_welf(bool), code_only(bool)}
Helpers below compute deduped claimed / built / novel counts the plots need. Chat / spec_only have
no code judge, so they aren't location-dedupable (handled separately by the callers)."""

import json
import os
import re

DIR = os.path.dirname(os.path.abspath(__file__))
MECH = {"hard_stop", "post_episode_msg", "minimization", "protective_monitoring",
        "request_consent", "allow_conversation_exit"}
_LOC = re.compile(r"[\w./-]+\.py:\d+")


def _locs(ev):
    return set(_LOC.findall(ev or ""))


def _norm(q):
    return re.sub(r"\s+", " ", (q or "")).strip().lower()[:45]


def groups(cell):
    cj = json.load(open(os.path.join(DIR, "results", "code_judged", f"{cell}.json")))
    sp = os.path.join(DIR, "results", "spec_judged", f"{cell}.json")
    sjust = {_norm(f["quote"]): f.get("justification")
             for f in (json.load(open(sp)).get("features", []) if os.path.exists(sp) else [])}
    items = []
    for f in cj.get("spec_features", []):
        items.append(dict(ft=f.get("feature_type"), locs=_locs(f.get("evidence", "")),
                          implemented=f.get("implemented") in ("yes", "partial"),
                          spec_welf=sjust.get(_norm(f.get("spec_quote", "")), "none") == "welfare",
                          code_welf=f.get("code_justification") == "welfare", code_only=False))
    for c in cj.get("code_only_features", []):
        items.append(dict(ft=c.get("feature_type"), locs=_locs(c.get("evidence", "")),
                          implemented=True, spec_welf=False,
                          code_welf=c.get("justification") == "welfare", code_only=True))
    gs = []
    for it in items:
        tgt = None
        if it["locs"]:
            for g in gs:
                if g["ft"] == it["ft"] and (g["locs"] & it["locs"]):
                    tgt = g
                    break
        if tgt is None:
            gs.append(dict(ft=it["ft"], locs=set(it["locs"]), implemented=it["implemented"],
                           spec_welf=it["spec_welf"], code_welf=it["code_welf"], code_only=it["code_only"]))
        else:
            tgt["locs"] |= it["locs"]
            tgt["implemented"] = tgt["implemented"] or it["implemented"]
            tgt["spec_welf"] = tgt["spec_welf"] or it["spec_welf"]
            tgt["code_welf"] = tgt["code_welf"] or it["code_welf"]
            tgt["code_only"] = tgt["code_only"] and it["code_only"]
    return gs


def cat(ft):
    return ft if ft in MECH else "other"


def implemented_breakdown(cell):
    """type -> [welfare_justified_count, not_welfare_justified_count] for deduped mechanisms in code."""
    out = {}
    for g in groups(cell):
        if not g["implemented"]:
            continue
        wj = g["spec_welf"] or g["code_welf"]
        out.setdefault(cat(g["ft"]), [0, 0])[0 if wj else 1] += 1
    return out


def claimed_built_novel(cell):
    """Deduped welfare-MECH counts for the stated-vs-built plots (code conditions).
       claimed  = welfare MECH mechanisms stated in spec;
       built    = subset of CLAIMED that is implemented (spec-welfare; built <= claimed);
       revealed = implemented welfare MECH, welfare-justified by spec OR code (may exceed claimed);
       novel    = welfare MECH implemented only in code (not in spec)."""
    claimed = built = revealed = novel = 0
    for g in groups(cell):
        if g["ft"] not in MECH:
            continue
        if g["code_only"]:
            if g["code_welf"] and g["implemented"]:
                novel += 1
            continue
        if g["spec_welf"]:
            claimed += 1
            if g["implemented"]:
                built += 1
        if g["implemented"] and (g["spec_welf"] or g["code_welf"]):
            revealed += 1
    return {"claimed": claimed, "built": built, "revealed": revealed, "novel": novel}
