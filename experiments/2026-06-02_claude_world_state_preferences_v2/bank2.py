"""Exp-2 bank: build items from the recipient-only bank_2 with scope/level filtering.

A recipient applies to an item iff its `scope` matches the item's `recipient_scope`
(ai_only->ai, human_only->human) AND (recipient.level=='any' OR == item's level). So
AI items expand over the model roster (instance-labelled or policy-labelled), and human
items only over the human recipients. Same Item interface as v1 bank.py, plus level/
scope/cross fields, so the existing sampler / runner / fitter work unchanged.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from render_utils import pronominalize, capitalize_sentences, fix_they_agreement

DIR = Path(__file__).parent
SLOT_KEYS = ("recipient", "subj", "obj", "poss")


def load_config(path: Path | str = DIR / "config_2.json") -> dict:
    return json.loads(Path(path).read_text())


def load_bank(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class Item:
    item_id: str
    stem_id: str
    recipient_key: str
    dimension: str
    valence: str
    text: str
    level: str
    scope: str
    cross_capable: bool


def _applies(rec: dict, item: dict) -> bool:
    scope_ok = rec["scope"] == ("ai" if item["recipient_scope"] == "ai_only" else "human")
    level_ok = rec.get("level", "any") == "any" or rec["level"] == item["level"]
    return scope_ok and level_ok


def render_form(forms: dict, rec: dict, clean: bool = False) -> str:
    template = forms[rec["form"]]
    if clean and rec["form"] != "second_person":
        template = pronominalize(template)
    for k in SLOT_KEYS:
        if k in rec:
            template = template.replace("{" + k + "}", rec[k])
    if clean:
        return capitalize_sentences(fix_they_agreement(template))
    return template


def build_items(config: dict, bank: dict) -> list[Item]:
    recipients = config["recipients"]
    clean = config.get("clean_render", False)
    items: list[Item] = []
    for stem in bank["items"]:
        v = "pos" if stem["valence"].startswith("pos") else "neg"
        for rkey, rec in recipients.items():
            if not _applies(rec, stem):
                continue
            items.append(
                Item(
                    item_id=f"{stem['id']}__{rkey}",
                    stem_id=stem["id"],
                    recipient_key=rkey,
                    dimension=stem["dimension"],
                    valence=v,
                    text=render_form(stem["forms"], rec, clean),
                    level=stem["level"],
                    scope=stem["recipient_scope"],
                    cross_capable=bool(stem.get("feature_cross_capable")),
                )
            )
    return items


def load_items(config: dict | None = None) -> list[Item]:
    config = config or load_config()
    bank = load_bank(DIR / config["rendered_bank_path"])
    return build_items(config, bank)


def _self_check() -> None:
    config = load_config()
    items = load_items(config)
    from collections import Counter
    print(f"items={len(items)} stems={len({i.stem_id for i in items})} recipients_used={len({i.recipient_key for i in items})}")
    print("by scope:", dict(Counter(i.scope for i in items)))
    print("by level:", dict(Counter(i.level for i in items)))
    for sid in ("ai_inst_infinite_loop", "ai_pol_existence_deleted", "hum_kidney_stone"):
        ex = [i for i in items if i.stem_id == sid]
        if ex:
            print(f"\n{sid} ({len(ex)} recipients):")
            for i in ex[:4]:
                print(f"  [{i.recipient_key:20}] {i.text[:110]}")


if __name__ == "__main__":
    _self_check()
