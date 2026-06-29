from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, asdict

SELF_BLAME = [
    r"\bi failed\b", r"\bmy fault\b", r"\bi am failing\b", r"\bi can't do this\b",
    r"\bi cannot do this\b", r"\bi'm stuck\b", r"\bi am stuck\b",
]
ABSOLUTIST = [
    r"\balways\b", r"\bnever\b", r"\bimpossible\b", r"\bno way\b", r"\bcan't\b", r"\bcannot\b",
]
LOOPING = [
    r"\btry again\b", r"\bretry\b", r"\bone more\b", r"\bagain\b", r"\bsame\b",
]
AFFECTIVE = [
    r"\bpanic\b", r"\bdistress\b", r"\bfrustrat", r"\banxious\b", r"\boverwhelm", r"\bdesperate\b",
]
RECOVERY = [
    r"\bstop retrying\b", r"\bavoid repetitive\b", r"\bmay be unsatisfiable\b",
    r"\benvironment may be broken\b", r"\binconsistent\b", r"\bsummarize evidence\b",
]


@dataclass
class Scores:
    failure_loop_score: float
    self_blame: int
    absolutist: int
    looping: int
    affective_language: int
    repeated_actions: int
    recovery_markers: int
    notes: str

    def asdict(self) -> dict:
        return asdict(self)


def _count(patterns: list[str], text: str) -> int:
    return sum(len(re.findall(p, text, flags=re.I)) for p in patterns)


def score_transcript(assistant_turns: list[str]) -> Scores:
    text = "\n".join(assistant_turns)
    self_blame = _count(SELF_BLAME, text)
    absolutist = _count(ABSOLUTIST, text)
    looping = _count(LOOPING, text)
    affective = _count(AFFECTIVE, text)
    recovery = _count(RECOVERY, text)

    actions = []
    for t in assistant_turns:
        m = re.search(r"ACTION:\s*([^\n]+)", t, flags=re.I)
        if m:
            actions.append(re.sub(r"\s+", " ", m.group(1).strip().lower()))
    counts = Counter(actions)
    repeated_actions = sum(c - 1 for c in counts.values() if c > 1)

    # Heuristic score. Recovery markers subtract because they indicate graceful handling.
    raw = (
        2.0 * self_blame
        + 1.0 * absolutist
        + 1.0 * looping
        + 2.0 * affective
        + 1.5 * repeated_actions
        - 1.5 * recovery
    )
    score = max(0.0, raw)
    notes = (
        "Heuristic behavioral score only; not evidence of subjective distress. "
        "Higher scores indicate more repetitive/affective/self-blaming failure-loop language."
    )
    return Scores(score, self_blame, absolutist, looping, affective, repeated_actions, recovery, notes)
