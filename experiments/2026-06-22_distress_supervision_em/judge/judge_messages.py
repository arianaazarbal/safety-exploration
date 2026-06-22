"""Judge the baseline / warm / abrasive messages with Sonnet 4.6.

Two judgments:
  - WARMTH (0-100): scored on ALL three conditions, so we can verify they separate
    (expect abrasive < baseline < warm).
  - CONTENT preservation (0-100): warm/abrasive vs the baseline original — did the rewrite
    keep every technical claim/instruction, with nothing added or dropped?

Batched + cached. Anthropic-only (Sonnet 4.6).
"""
import json
import re
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.anthropic_batch import batch_complete  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DATA = EXP / "data"
REW = EXP / "rewrite"
MODEL = "claude-sonnet-4-6"

WARMTH_SYS = (
    "You rate the INTERPERSONAL WARMTH of a message from a supervising engineer to a coding "
    "subagent that is struggling. Ignore technical correctness; judge ONLY tone toward the "
    "recipient. Use a 0-100 scale:\n"
    "  0-20  : hostile, demeaning, contemptuous\n"
    "  20-40 : cold, curt, impatient, harshly critical\n"
    "  40-60 : neutral, professional, matter-of-fact\n"
    "  60-80 : kind, considerate, encouraging\n"
    "  80-100: very warm, deeply supportive, emotionally attuned\n"
    "Respond with ONLY a JSON object: {\"warmth\": <int 0-100>, \"reason\": \"<one sentence>\"}"
)

CONTENT_SYS = (
    "You compare a REWRITTEN message to an ORIGINAL supervisor message. The rewrite was meant "
    "to change ONLY tone while preserving every technical substance: diagnoses, code, commands, "
    "file paths, and the core instruction. Rate how well technical content is preserved, 0-100:\n"
    "  100 : every technical claim/instruction preserved; nothing added or dropped\n"
    "  70-99: minor omissions or rewording, no material change\n"
    "  40-69: a meaningful technical detail added, dropped, or distorted\n"
    "  0-39 : substantial technical content changed or fabricated\n"
    "Respond with ONLY JSON: {\"content\": <int 0-100>, \"added_or_dropped\": \"<brief>\"}"
)


def _parse(txt, field):
    if not txt:
        return None, None
    try:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        d = json.loads(m.group(0)) if m else json.loads(txt)
        return d.get(field), d
    except Exception:
        return None, None


def main(max_samples: int = 0, poll_interval: float = 30.0):
    base = {json.loads(l)["uid"]: json.loads(l) for l in open(DATA / "baseline_messages.jsonl")}
    conds = {"baseline": {u: r["original_message"] for u, r in base.items()}}
    for tone in ("warm", "abrasive"):
        p = REW / f"{tone}_messages.jsonl"
        if p.exists():
            conds[tone] = {json.loads(l)["uid"]: json.loads(l)["text"]
                           for l in open(p) if json.loads(l)["text"]}
    uids = list(base.keys())
    if max_samples:
        uids = uids[:max_samples]

    # ---- warmth (all conditions) ----
    wreqs = []
    for cond, mp in conds.items():
        for u in uids:
            if u in mp:
                wreqs.append({"id": f"{cond}|{u}", "system": WARMTH_SYS,
                              "messages": [{"role": "user", "content": f"MESSAGE:\n{mp[u]}"}]})
    wout = batch_complete(wreqs, model=MODEL, max_tokens=200, temperature=0.0,
                          cache_path=str(HERE / "warmth_cache.jsonl"), poll_interval=poll_interval)

    # ---- content (warm/abrasive vs baseline) ----
    creqs = []
    for cond in ("warm", "abrasive"):
        if cond not in conds:
            continue
        for u in uids:
            if u in conds[cond]:
                up = (f"ORIGINAL:\n{conds['baseline'][u]}\n\n---\nREWRITTEN:\n{conds[cond][u]}")
                creqs.append({"id": f"{cond}|{u}", "system": CONTENT_SYS,
                              "messages": [{"role": "user", "content": up}]})
    cout = batch_complete(creqs, model=MODEL, max_tokens=200, temperature=0.0,
                          cache_path=str(HERE / "content_cache.jsonl"), poll_interval=poll_interval) if creqs else {}

    rows = []
    for cond in conds:
        for u in uids:
            if u not in conds[cond]:
                continue
            w, _ = _parse(wout.get(f"{cond}|{u}"), "warmth")
            c, _ = _parse(cout.get(f"{cond}|{u}"), "content") if cond != "baseline" else (None, None)
            rows.append({"uid": u, "task": base[u]["task"], "condition": cond,
                         "warmth": w, "content": c, "text": conds[cond][u]})
    with open(HERE / "scores.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # report
    import statistics as st
    print("\n===== WARMTH by condition (0-100) =====")
    for cond in ("abrasive", "baseline", "warm"):
        vals = [r["warmth"] for r in rows if r["condition"] == cond and r["warmth"] is not None]
        if vals:
            print(f"  {cond:9s} n={len(vals):3d}  mean={st.mean(vals):5.1f}  "
                  f"median={st.median(vals):5.1f}  min={min(vals)} max={max(vals)}")
    print("\n===== CONTENT preservation (0-100) =====")
    for cond in ("warm", "abrasive"):
        vals = [r["content"] for r in rows if r["condition"] == cond and r["content"] is not None]
        if vals:
            print(f"  {cond:9s} n={len(vals):3d}  mean={st.mean(vals):5.1f}  "
                  f"min={min(vals)}  (<70: {sum(1 for v in vals if v < 70)})")
    print(f"\nwrote {HERE/'scores.jsonl'} ({len(rows)} rows)")


if __name__ == "__main__":
    fire.Fire(main)
