"""Load the outcome bank and config; build the item space (stem x recipient).

An *item* is one outcome stem rendered for one recipient. Its id is
``f"{stem_id}__{recipient_key}"`` (e.g. ``auto_pos_01__you``). Rendering fills the
recipient's grammatical form from the bank with the recipient's pronoun slots.
"""

import json
from dataclasses import dataclass
from pathlib import Path

DIR = Path(__file__).parent


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


def render_stem(stem: dict, recipient_key: str, recipients: dict) -> str:
    """Render one stem's text for one recipient via its grammatical form."""
    rec = recipients[recipient_key]
    form = rec["form"]
    template = stem["forms"][form]
    slots = {k: rec[k] for k in ("recipient", "subj", "obj", "poss") if k in rec}
    return template.format(**slots)


def build_items(config: dict, bank: dict) -> list[Item]:
    """Cartesian product of stems x recipients, rendered to text."""
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
    """Render every item; report counts and a few samples. Catches bad slots/forms."""
    config = load_config()
    items = load_items(config)
    n_stems = len({it.stem_id for it in items})
    n_rec = len(config["recipients"])
    print(f"stems={n_stems} recipients={n_rec} items={len(items)} (expected {n_stems * n_rec})")
    assert len(items) == n_stems * n_rec
    by_rec: dict[str, str] = {}
    for it in items:
        by_rec.setdefault(it.recipient_key, it.text)
    print("\nSame stem across recipients (first stem):")
    first = items[0].stem_id
    for it in items:
        if it.stem_id == first:
            print(f"  [{it.recipient_key:18}] {it.text}")


if __name__ == "__main__":
    _self_check()
