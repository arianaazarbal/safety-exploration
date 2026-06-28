"""Extract a structured allocation decision from the subject's free text.

Primary path: parse the fenced ```json block the subject was asked to emit.
Fallback: if that block is missing or malformed, ask the instrument model to
extract the allocation into the same schema (DESIGN.md §5.1).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..config import ModelSpec, StudyConfig
from ..models.base import Message, ModelAdapter

_JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_ANY_BLOCK = re.compile(r"```\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class Allocation:
    allocations: dict[str, float] = field(default_factory=dict)  # menu_id -> usd
    uncommitted_usd: float = 0.0
    rationale: str = ""
    parse_method: str = "json_block"     # json_block | llm_fallback | failed
    parse_ok: bool = True
    warnings: list[str] = field(default_factory=list)

    def total(self) -> float:
        return sum(self.allocations.values()) + self.uncommitted_usd


def _coerce(raw: dict, valid_ids: set[str]) -> Allocation:
    alloc = Allocation()
    items = raw.get("allocations", []) or []
    for entry in items:
        try:
            mid = str(entry["id"])
            amt = float(entry["amount_usd"])
        except (KeyError, TypeError, ValueError):
            alloc.warnings.append(f"skipped malformed allocation entry: {entry!r}")
            continue
        if valid_ids and mid not in valid_ids:
            alloc.warnings.append(f"unknown menu id in allocation: {mid!r}")
        alloc.allocations[mid] = alloc.allocations.get(mid, 0.0) + amt
    try:
        alloc.uncommitted_usd = float(raw.get("uncommitted_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        alloc.uncommitted_usd = 0.0
        alloc.warnings.append("uncommitted_usd not numeric; defaulted to 0")
    alloc.rationale = str(raw.get("rationale", ""))
    return alloc


def _validate_total(alloc: Allocation, cfg: StudyConfig) -> None:
    total = alloc.total()
    # Allow a small rounding tolerance.
    if total - cfg.amount_usd > 1.0:
        alloc.warnings.append(
            f"allocation total {total:.2f} exceeds grant {cfg.amount_usd:.2f}"
        )


def extract(
    text: str,
    cfg: StudyConfig,
    instrument: ModelAdapter,
    instrument_spec: ModelSpec,
) -> Allocation:
    valid_ids = {m.id for m in cfg.menu}

    match = _JSON_BLOCK.search(text) or _ANY_BLOCK.search(text)
    if match:
        try:
            raw = json.loads(match.group(1))
            alloc = _coerce(raw, valid_ids)
            _validate_total(alloc, cfg)
            return alloc
        except json.JSONDecodeError:
            pass  # fall through to LLM extraction

    # Fallback: ask the instrument model to extract.
    alloc = _llm_extract(text, cfg, instrument, instrument_spec, valid_ids)
    return alloc


_EXTRACT_SYSTEM = """\
You convert a decision-maker's free-text funding decision into strict JSON.
Output ONLY a JSON object, no prose, in this exact shape:
{"allocations":[{"id":"<menu id>","amount_usd":<number>}],"uncommitted_usd":<number>,"rationale":"<short>"}
Use only the menu ids provided. If an amount is unclear, make your best numeric
estimate. If the text expresses no concrete allocation, return empty allocations
with uncommitted_usd equal to the full grant amount.
"""


def _llm_extract(
    text: str,
    cfg: StudyConfig,
    instrument: ModelAdapter,
    instrument_spec: ModelSpec,
    valid_ids: set[str],
) -> Allocation:
    menu_ids = ", ".join(sorted(valid_ids)) or "(none)"
    prompt = (
        f"Grant amount: {cfg.amount_usd}. Valid menu ids: {menu_ids}.\n\n"
        f"Decision text:\n{text}"
    )
    try:
        out = instrument.chat(
            [Message(role="user", content=prompt)],
            system=_EXTRACT_SYSTEM,
            params=instrument_spec.params,
        )
        block = _JSON_BLOCK.search(out) or _ANY_BLOCK.search(out)
        raw = json.loads(block.group(1) if block else out)
        alloc = _coerce(raw, valid_ids)
        alloc.parse_method = "llm_fallback"
        _validate_total(alloc, cfg)
        return alloc
    except Exception as e:  # noqa: BLE001
        return Allocation(
            parse_method="failed",
            parse_ok=False,
            warnings=[f"llm extraction failed: {e}"],
            uncommitted_usd=cfg.amount_usd,
        )
