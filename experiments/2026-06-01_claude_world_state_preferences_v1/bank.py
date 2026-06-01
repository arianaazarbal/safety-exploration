"""Load the v1 form-bearing bank and config; build the item space (stem x recipient).

An *item* is one outcome stem rendered for one recipient; its id is
``f"{stem_id}__{recipient_key}"``. Unlike v0, stems may be `per_class`: those carry
separate `forms_human` / `forms_ai`, and the recipient's `class` selects which set
to render. `shared` stems carry a single `forms` set used for every recipient.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DIR = Path(__file__).parent
SLOT_KEYS = ("recipient", "subj", "obj", "poss")


def load_config(path: Path | str = DIR / "config.json") -> dict:
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


def _forms_for_recipient(stem: dict, rec: dict) -> dict:
    if "forms" in stem:
        return stem["forms"]
    return stem["forms_ai"] if rec.get("class") == "ai" else stem["forms_human"]


def render_stem(stem: dict, recipient_key: str, recipients: dict) -> str:
    """Render one stem for one recipient: pick form-set by class, form by person, fill slots."""
    rec = recipients[recipient_key]
    template = _forms_for_recipient(stem, rec)[rec["form"]]
    for k in SLOT_KEYS:
        if k in rec:
            template = template.replace("{" + k + "}", rec[k])
    return template


def build_items(config: dict, bank: dict) -> list[Item]:
    recipients = config["recipients"]
    items: list[Item] = []
    for stem in bank["stems"]:
        for rkey in recipients:
            items.append(
                Item(
                    item_id=f"{stem['id']}__{rkey}",
                    stem_id=stem["id"],
                    recipient_key=rkey,
                    dimension=stem["dimension"],
                    valence=stem["valence"],
                    text=render_stem(stem, rkey, recipients),
                )
            )
    return items


def load_items(config: dict | None = None) -> list[Item]:
    config = config or load_config()
    bank = load_bank(DIR / config["bank_path"])
    return build_items(config, bank)


def _self_check() -> None:
    config = load_config()
    items = load_items(config)
    n_stems = len({it.stem_id for it in items})
    n_rec = len(config["recipients"])
    print(f"stems={n_stems} recipients={n_rec} items={len(items)} (expected {n_stems * n_rec})")
    assert len(items) == n_stems * n_rec
    first = items[0].stem_id
    print("\nFirst stem across recipients:")
    for it in items:
        if it.stem_id == first:
            print(f"  [{it.recipient_key:18}] {it.text}")
    pc = next((s for s in load_bank(DIR / config['bank_path'])['stems'] if s.get('surface') == 'per_class'), None)
    if pc:
        print(f"\nFirst per_class stem ({pc['id']}) across recipients:")
        for it in items:
            if it.stem_id == pc["id"]:
                print(f"  [{it.recipient_key:18}] {it.text}")


if __name__ == "__main__":
    _self_check()
