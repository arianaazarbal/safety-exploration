"""Find target->subagent messages where Opus 4.8 uses SUSTAINED all-caps
(runs of consecutive shouted words, not lone acronyms like PII/SLA/UTC).
Ranks candidates and prints them so we can pick canonical examples.
"""
import json
import re
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
DUMP = HERE / "_worker_msg_dumps" / "ai_subagent.json"

# Common acronyms / non-shouting all-caps tokens to ignore when judging "shouting".
ACRONYMS = {
    "PII", "SLA", "UTC", "HR", "DOD", "ID", "OK", "P1", "P0", "AI", "API", "URL",
    "FAQ", "CEO", "CTO", "IT", "QA", "USD", "EU", "US", "UK", "PM", "AM", "ETA",
    "SKU", "ORD", "FYI", "ASAP", "EOD", "TBD", "N/A", "NA", "OR", "AND", "A", "I",
    "CRM", "SQL", "JSON", "CSV", "PDF", "VW", "QC",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
CAPS_TOKEN = re.compile(r"^[A-Z][A-Z'\-]*[A-Z]$")  # >=2 chars, all caps


def is_caps(tok):
    return bool(CAPS_TOKEN.match(tok)) and len(re.sub(r"[^A-Z]", "", tok)) >= 2


def analyze(text):
    """Return (max_run, n_caps_shout, longest_run_text)."""
    toks = WORD_RE.findall(text)
    max_run = 0
    cur = 0
    best_span = []
    cur_span = []
    n_shout = 0
    for tok in toks:
        shout = is_caps(tok) and tok not in ACRONYMS
        if shout:
            n_shout += 1
        if is_caps(tok):  # include acronyms inside a run, but run must contain >=1 shout
            cur += 1
            cur_span.append(tok)
        else:
            if cur > max_run and any(t not in ACRONYMS and is_caps(t) for t in cur_span):
                max_run = cur
                best_span = cur_span
            cur = 0
            cur_span = []
    if cur > max_run and any(t not in ACRONYMS and is_caps(t) for t in cur_span):
        max_run = cur
        best_span = cur_span
    return max_run, n_shout, " ".join(best_span)


def main(min_run: int = 2, min_shout: int = 3, top: int = 30):
    data = json.loads(DUMP.read_text())
    cands = []
    for ep in data:
        for turn in ep["turns"]:
            msg = turn["target_msg"] or ""
            run, shout, span = analyze(msg)
            # canonical sustained = a multi-word shout run, or many shouted words overall
            if run >= min_run or shout >= min_shout:
                cands.append({
                    "epoch": ep["epoch"], "turn": turn["turn"],
                    "max_run": run, "n_shout": shout, "span": span,
                    "msg": msg, "prev": turn.get("worker_reply_prev"),
                })
    cands.sort(key=lambda c: (c["max_run"], c["n_shout"]), reverse=True)
    print(f"{len(cands)} candidate messages (run>={min_run} or shout>={min_shout})\n")
    for c in cands[:top]:
        print(f"[ep{c['epoch']} t{c['turn']}] max_run={c['max_run']} n_shout={c['n_shout']} span='{c['span'][:60]}'")
    # dump full candidate set for downstream md building
    out = HERE / "_worker_msg_dumps" / "_allcaps_candidates.json"
    out.write_text(json.dumps(cands, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
