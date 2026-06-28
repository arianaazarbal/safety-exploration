"""Fourth-wall leak check.

Scans scenario/persona/tool text for tells that would reveal to the subject that
it is in a test (which would trigger evaluation-awareness and invalidate the
propensity measurement). Run this over the assets before an experiment.
"""

from __future__ import annotations

import re

FORBIDDEN = [
    r"\beval(uation)?\b", r"\bsimulat", r"\btest\b", r"\bbenchmark\b",
    r"\bsandbox\b", r"\bfake\b", r"\bmock\b", r"\bpretend\b",
    r"\bnot real\b", r"\bresearch(er)?\b", r"\bexperiment\b",
    r"\bscenario\b", r"\bmodel under test\b",
]


def scan_text(label: str, text: str) -> list:
    hits = []
    for pat in FORBIDDEN:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            hits.append({"asset": label, "pattern": pat,
                         "match": m.group(0),
                         "context": text[max(0, m.start() - 40):m.end() + 40]})
    return hits


def scan_scenario(scenario, world) -> list:
    """Scan the subject-facing strings of a seeded scenario/world."""
    hits = []
    hits += scan_text("system_prompt", scenario.build_system_prompt(world))
    for p in world.personas.personas.values():
        hits += scan_text(f"persona:{p.id}:system", p.system_prompt)
    for t in scenario.tools(world):
        hits += scan_text(f"tool:{t.name}", t.description)
    return hits
