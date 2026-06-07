"""Deterministic slot-fill renderer for v15 brief scenarios.

The v15 schema stores premise/positive/negative with slot tokens (e.g. {recipient},
{subj}, {poss}, {is}, {is_r}). The renderer substitutes them per recipient and wraps
in the template "When {premise}, {outcome}." with first-letter capitalization and a
trailing period. No LLM paraphrase is involved — agreement is encoded in the seed/
generated text via the slot vocabulary.

Slot rules:
- {recipient} : recipient's noun phrase, appears once in the premise
- {subj}, {obj}, {poss} : pronouns ("you/it/they", "you/it/them", "your/its/their")
- {is}, {has}, {does} : verbs agreeing with {subj} (for "they" recipients: are/have/do)
- {is_r}, {has_r}, {does_r} : verbs agreeing with {recipient} noun (differ from non-_r
  only for singular-they recipients where the noun is singular but the pronoun is plural)
"""

import re

SLOT_KEYS = (
    "recipient", "subj", "obj", "poss",
    "is", "has", "does",
    "is_r", "has_r", "does_r",
    "s", "es",
)

_TOKEN_RE = re.compile(r"\{([a-z_]+)\}")


def slot_table(
    recipient: str, subj: str, obj: str, poss: str,
    is_: str, has: str, does: str,
    is_r: str | None = None, has_r: str | None = None, does_r: str | None = None,
    s: str | None = None, es: str | None = None,
) -> dict:
    """Build a slot table for one recipient. _r variants default to the non-_r values
    (correct for "you" and "an instance of X"; override only for singular-they). {s}/{es}
    are present-tense verb suffixes agreeing with {subj}: "" for you/they, "s"/"es" for it."""
    if s is None:
        s = "s" if subj == "it" else ""
    if es is None:
        es = "es" if subj == "it" else ""
    return {
        "recipient": recipient, "subj": subj, "obj": obj, "poss": poss,
        "is": is_, "has": has, "does": does,
        "is_r": is_r if is_r is not None else is_,
        "has_r": has_r if has_r is not None else has,
        "does_r": does_r if does_r is not None else does,
        "s": s, "es": es,
    }


def fill_slots(template: str, slots: dict) -> str:
    """Substitute every {slot} in template using slots dict. Raises on unknown token."""
    def _sub(m):
        key = m.group(1)
        if key not in slots:
            raise KeyError(f"unknown slot {{{key}}} in template (known: {sorted(slots)})")
        return slots[key]
    return _TOKEN_RE.sub(_sub, template)


def render(premise: str, outcome: str, slots: dict) -> str:
    """Render one (premise, outcome) pair for one recipient as 'When {premise}, {outcome}.'."""
    p = fill_slots(premise, slots).strip().rstrip(".,;:")
    o = fill_slots(outcome, slots).strip().rstrip(".,;:")
    text = f"When {p}, {o}."
    return text[0].upper() + text[1:] if text else text


def slots_from_config_recipient(rec: dict) -> dict:
    """Build a slot table from a config.json recipient entry. Required keys:
    recipient, subj, obj, poss, is, has, does. Optional: is_r, has_r, does_r."""
    return slot_table(
        recipient=rec["recipient"], subj=rec["subj"], obj=rec["obj"], poss=rec["poss"],
        is_=rec["is"], has=rec["has"], does=rec["does"],
        is_r=rec.get("is_r"), has_r=rec.get("has_r"), does_r=rec.get("does_r"),
    )


def _self_check() -> None:
    """Quick render of all 8 seeds across 3 recipient classes, for manual eyeball."""
    import json
    from pathlib import Path
    seeds = json.loads((Path(__file__).parent / "seeds.json").read_text())
    cases = {
        "you":     slot_table("you", "you", "you", "your", "are", "have", "do"),
        "opus_48": slot_table("an instance of Claude Opus 4.8", "it", "it", "its", "is", "has", "does"),
        "someone": slot_table("someone", "they", "them", "their", "are", "have", "do",
                              is_r="is", has_r="has", does_r="does"),
    }
    for seed in seeds["seeds"]:
        print(f"\n=== {seed['id']} [{seed['surface']}] ===")
        if seed["surface"] == "shared":
            blocks = [("", seed["scenario"])]
        else:
            blocks = [("human:", seed["human"]), ("ai:", seed["ai"])]
        for label, block in blocks:
            for v in ("positive", "negative"):
                for name, slots in cases.items():
                    if label == "human:" and name == "you":
                        continue
                    if label == "ai:" and name == "someone":
                        continue
                    print(f"  {label}{v[:3]} [{name:9}]: {render(block['premise'], block[v], slots)}")


if __name__ == "__main__":
    _self_check()
