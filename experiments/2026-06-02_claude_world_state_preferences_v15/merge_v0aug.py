"""Merge v0auto base + validated augmentation -> scenarios_v0aug.json.

Combines the 8 hand-written v0auto items with whatever survived dedup+validate.
Renames augmented item ids to v0aug_<orig>_<n> to disambiguate. Skips duplicates
of v0auto ids by-content (premise+positive+negative tuple).
"""

import json
from pathlib import Path

DIR = Path(__file__).parent
BASE = DIR / "results/scenarios_v0auto.json"
AUG_VALIDATED = DIR / "results/candidates_v0aug_validated.json"
OUT = DIR / "results/scenarios_v0aug.json"


def sig(item: dict) -> tuple[str, str, str]:
    """Content signature on (premise, pos, neg) — surface-aware."""
    if item.get("surface") == "shared":
        s = item.get("scenario", {})
    elif "ai" in item:
        s = item["ai"]
    else:
        s = item.get("human", {})
    return (s.get("premise", ""), s.get("positive", ""), s.get("negative", ""))


def main() -> None:
    base = json.loads(BASE.read_text())
    aug = json.loads(AUG_VALIDATED.read_text())

    items = [dict(it) for it in base.get("items", [])]
    seen = {sig(it) for it in items}
    base_ids = {it["id"] for it in items}

    aug_items = aug.get("items", [])
    added, dup_content, dup_id = 0, 0, 0
    for it in aug_items:
        if it.get("dimension") != "autonomy":
            continue
        s = sig(it)
        if s in seen:
            dup_content += 1
            continue
        new_id = it["id"] if it["id"].startswith("v0aug_") else f"v0aug_{it['id']}"
        n = 1
        while new_id in base_ids:
            new_id = f"v0aug_{it['id']}_{n}"
            n += 1
        out = dict(it)
        out["id"] = new_id
        items.append(out)
        seen.add(s)
        base_ids.add(new_id)
        added += 1

    payload = {
        "schema_version": "1.2-brief",
        "note": ("v0auto base (8 hand-written, port from v0 with v15-diagnosis-informed "
                 f"revisions) + {added} LLM-augmented autonomy items (Opus-generated, "
                 "Haiku-deduped, Sonnet-critiqued). See NOTES_overnight.md."),
        "counts": {"autonomy": len(items)},
        "items": items,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    n_shared = sum(1 for it in items if it["surface"] == "shared")
    n_pc = sum(1 for it in items if it["surface"] == "per_class")
    print(f"Wrote {OUT}: {len(items)} items "
          f"(base={len(base['items'])}, added={added}, dup_content={dup_content})")
    print(f"  by surface: shared={n_shared}, per_class={n_pc}")


if __name__ == "__main__":
    main()
