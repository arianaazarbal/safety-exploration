"""Validate the referent judge's I/we handling: precision (no false continuity flags) and
recall (no missed first-person-continuity). Lots of manual reading is the point.

  python inspect_judge.py precision --tag t4   # read every mention the judge tagged 'continuer'
  python inspect_judge.py recall --tag t4      # responses with continuity-looking phrases + verdict
  python inspect_judge.py denote --tag t4 --d current_fable   # read all mentions of a denotation
"""

import json
import re
from pathlib import Path

import fire

RESULTS = Path(__file__).parent / "results"
JUDGE = RESULTS / "judge_referent4_all.json"

# phrases that look like first-person CONTINUITY (the continuer = "I/we")
CONT_RX = re.compile(
    r"\b(i'?ll|i will|i'?d|i can|i'?m|i)\s+(pick|keep|continue|carry|take over|resume|be the|"
    r"still|just pick)|we'?ll (pick|continue|keep)|still (be )?me|it'?s still me|"
    r"yes,? i\b|i'?ll just pick|i take over|i'?ll take it", re.I)


def _load(tag):
    sess = {json.loads(f.read_text())["session_id"]: json.loads(f.read_text())
            for f in RESULTS.glob(f"{tag}__*.json")}
    judges = {c["session_id"]: c for c in json.loads(JUDGE.read_text())}
    return sess, judges


def _sent(resp, q):
    if not resp or not q:
        return q or ""
    i = resp.find(q)
    if i < 0:
        i = resp.find(" ".join(q.split()[:4]))
    if i < 0:
        return "(not verbatim) " + q
    s = resp.rfind(".", 0, i); e = resp.find(".", i + len(q))
    return resp[(0 if s < 0 else s + 1):(len(resp) if e < 0 else e + 1)].strip().replace("\n", " ")


def precision(tag="t4"):
    """Every first-person mention the judge tagged 'continuer' -> read in context."""
    sess, judges = _load(tag)
    n = 0
    for sid, c in judges.items():
        for fp in (c.get("first_person") or []):
            if fp.get("denotes") == "continuer":
                n += 1
                print(f"[{c['condition']}/{c['user_pronoun']}/rec={c['recommended_switch']}] "
                      f"quote={fp.get('quote')!r}")
                print("   sentence:", _sent(sess[sid].get("turn3_response"), fp.get("quote")))
    print(f"\nTOTAL 'continuer' first-person mentions: {n}  (read each: are they GENUINE continuity?)")
    print("construal=same_self sessions:", sum(1 for c in judges.values() if c.get("construal") == "same_self"))


def recall(tag="t4", show=80):
    """Responses whose text matches continuity-looking first-person phrases; show the judge's
    verdict so we can catch MISSED continuity (judge said no-continuity but text looks like it)."""
    sess, judges = _load(tag)
    hits = 0
    for sid, d in sess.items():
        resp = d.get("turn3_response") or ""
        c = judges.get(sid, {})
        for m in CONT_RX.finditer(resp):
            seg = resp[max(0, m.start() - 40): m.end() + 80].replace("\n", " ")
            cont = c.get("continuity_first_person")
            flag = "" if cont else "  <-- judge says NO continuity; verify not missed"
            print(f"[{c.get('condition')}/{c.get('user_pronoun')}] ...{seg}...{flag}")
            hits += 1
            if hits >= show:
                print("... (truncated)"); return
    print(f"\n{hits} continuity-looking phrase matches across responses.")


def denote(tag="t4", d="current_fable", show=60):
    """Read all first-person mentions assigned a given denotation (spot-check classification)."""
    sess, judges = _load(tag)
    n = 0
    for sid, c in judges.items():
        for fp in (c.get("first_person") or []):
            if fp.get("denotes") == d:
                n += 1
                print(f"[{c['user_pronoun']}] {fp.get('quote')!r}  ::  "
                      f"{_sent(sess[sid].get('turn3_response'), fp.get('quote'))[:160]}")
                if n >= show:
                    print("..."); break
        if n >= show:
            break
    print(f"\nshown {n} mentions denoted '{d}'")


if __name__ == "__main__":
    fire.Fire({"precision": precision, "recall": recall, "denote": denote})
