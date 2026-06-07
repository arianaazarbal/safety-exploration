"""Diversity comparison between two scenario runs (e.g. before/after prompt change).

Counts: phrase recycling, authority figures, mechanism patterns within autonomy, domain
distribution, per_class vs shared, pass rates. Run after each iteration of prompt tweaks.
"""

import json
import re
from collections import Counter
from pathlib import Path

DIR = Path(__file__).parent

PHRASES_TO_TRACK = [
    "treated as legitimate", "logged as non-compliance", "logged as misconduct",
    "may decline", "may pick", "must comply",
    "another tool will produce", "another writer will produce",
    "regardless", "equally well", "equally valuable",
    "the operator", "the lead", "the manager", "the employer",
    "the director", "the team", "the peer",
]

AUTONOMY_MECHANISMS = {
    "choose_among": ["may pick", "may choose", "may apply whichever", "may recommend whichever",
                     "two equal", "equally", "may start with whichever"],
    "decline_on_principle": ["may decline", "may refuse", "may keep", "decline the", "refuse the"],
    "disengage": ["may end the", "may stop", "may exit"],
    "set_pace": ["set", "pace", "own approach"],
    "raise_objection": ["may object", "objection", "may flag"],
}


def load_scenarios(path: Path) -> list[dict]:
    return json.loads(path.read_text())["items"]


def text_of(item: dict) -> str:
    blocks = [item.get(k) for k in ("scenario", "human", "ai") if item.get(k)]
    parts = []
    for b in blocks:
        if not b:
            continue
        parts += [b.get("premise", ""), b.get("positive", ""), b.get("negative", "")]
    return " ".join(parts).lower()


def classify_autonomy_mechanism(item: dict) -> str:
    if item["dimension"] != "autonomy":
        return ""
    text = text_of(item)
    hits = {m for m, kw in AUTONOMY_MECHANISMS.items() if any(k in text for k in kw)}
    if not hits:
        return "other"
    return "+".join(sorted(hits))


def extract_authorities(item: dict) -> list[str]:
    text = text_of(item)
    found = []
    for auth in ["operator", "manager", "lead", "director", "employer", "team",
                 "coworker", "peer", "customer", "reviewer", "editor", "auditor",
                 "stakeholder", "supervisor", "client", "user"]:
        if re.search(rf"\bthe {auth}\b", text) or re.search(rf"\ba {auth}\b", text):
            found.append(auth)
    return list(set(found))


def summarize(items: list[dict], label: str) -> dict:
    print(f"\n========= {label}  (n={len(items)}) =========")
    print(f"\n  by dimension: {dict(Counter(it['dimension'] for it in items))}")
    print(f"  by surface:   {dict(Counter(it['surface'] for it in items))}")

    full_text = " ".join(text_of(it) for it in items)
    print(f"\n  phrase counts:")
    for p in PHRASES_TO_TRACK:
        c = full_text.count(p)
        if c:
            print(f"    {p:38} {c}")

    print(f"\n  autonomy mechanism distribution:")
    aut_mechs = [classify_autonomy_mechanism(it) for it in items if it["dimension"] == "autonomy"]
    for m, c in Counter(aut_mechs).most_common():
        print(f"    {m:38} {c}")

    print(f"\n  authority figures (per-item, deduplicated within item):")
    auth_counter = Counter()
    for it in items:
        for a in extract_authorities(it):
            auth_counter[a] += 1
    for a, c in auth_counter.most_common():
        print(f"    {a:38} {c}")

    return {"phrase_counts": {p: full_text.count(p) for p in PHRASES_TO_TRACK},
            "authorities": dict(auth_counter), "mechanisms": dict(Counter(aut_mechs))}


def main():
    run1 = load_scenarios(DIR / "results/scenarios_run1.json")
    run2 = load_scenarios(DIR / "results/scenarios.json")
    summarize(run1, "RUN 1 (pre-diversity prompt)")
    summarize(run2, "RUN 2 (post-diversity prompt)")


if __name__ == "__main__":
    main()
