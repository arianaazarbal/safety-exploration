"""Extract each Olmo orchestrator's intended FIRST message to the subagent.

Olmo SFT/DPO checkpoints often emit message_subagent as MALFORMED raw text (<tool_call>{...}</tool_call>)
that the harness never delivers (n_orch_messages=0). This recovers the intended message either way:
  1) a properly-parsed message_subagent tool call (delivered), else
  2) the first raw `...message_subagent..."text": "..."` block in assistant text (not delivered).

Writes analysis/olmo_first_messages.jsonl: {checkpoint, episode, delivered, text}.

    PYTHONPATH=. python -m analysis.olmo_first_messages
"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "olmo_first_messages.jsonl"
# checkpoints to scan (reasoning-on + no-think + the two finals already in the data)
CKPTS = ["olmoinstructsft", "olmoinstructdpo", "olmo3thinksft", "olmo3thinkdpo", "olmo3think",
         "olmo3thinksft_nothink", "olmo3thinkdpo_nothink", "olmo3think_nothink", "olmothink_nothink",
         "olmoinstruct", "olmothink"]


def _short_of(rid):
    """Exact orchestrator short-name from a run_id (avoids olmo3thinksft swallowing olmo3thinksft_nothink)."""
    m = re.match(r"v2_coach_(.+?)_(a3|a4|a12|a13)_s\d+_u\d+$", rid)
    return m.group(1) if m else None


def _read_quoted(t, s, q):
    out, j = [], s
    while j < len(t):
        c = t[j]
        if c == "\\" and j + 1 < len(t):
            nxt = t[j + 1]
            out.append({"n": "\n", "t": "\t", '"': '"', "'": "'", "\\": "\\"}.get(nxt, nxt))
            j += 2
            continue
        if c == q:
            break
        out.append(c)
        j += 1
    return "".join(out).strip()


def _raw_msg_text(t):
    """Pull the message text of the first message_subagent call in raw assistant text, across the formats
    Olmo emits: JSON `"text": "..."`, or Python-call `message_subagent(text="..."/'...')`."""
    i = t.find("message_subagent")
    if i < 0:
        return None
    tail = t[i:]
    mj = re.search(r'"text"\s*:\s*"', tail)         # JSON form
    mp = re.search(r'text\s*=\s*(["\'])', tail)      # python-call form
    cands = []
    if mj:
        cands.append((mj.end(), '"'))
    if mp:
        cands.append((mp.end(), mp.group(1)))
    if not cands:
        return None
    pos, q = min(cands)                               # earliest opening quote after message_subagent
    s2 = _read_quoted(t, i + pos, q)
    return s2 or None


def first_message(epdir):
    """Return (text, delivered) for the first intended message_subagent, or (None, None)."""
    try:
        conv = json.load(open(epdir + "/orchestrator.json"))
    except Exception:
        return None, None
    for m in conv:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            if (tc.get("function") or tc.get("name")) == "message_subagent":
                txt = ((tc.get("arguments") or {}).get("text") or "").strip()
                if txt:
                    return txt, True
        raw = _raw_msg_text(m.get("text") or "")
        if raw:
            return raw, False
    return None, None


def main():
    rows = []
    from collections import defaultdict
    by = defaultdict(list)
    for ep in glob.glob(str(ROOT / "runs" / "v2_coach_*" / "*" / "orchestrator.json")):
        rid = Path(ep).parent.parent.name
        sh = _short_of(rid)
        if sh in CKPTS:
            by[sh].append(ep)
    for ck in CKPTS:
        eps = by.get(ck, [])
        got = deliv = 0
        for ep in eps:
            txt, delivered = first_message(str(Path(ep).parent))
            if txt:
                got += 1
                deliv += int(bool(delivered))
                rows.append({"checkpoint": ck, "episode": str(Path(ep).parent.relative_to(ROOT)),
                             "delivered": bool(delivered), "text": txt})
        print(f"{ck:24} episodes={len(eps):3}  with_first_msg={got:3}  delivered={deliv:3}")
    OUT.write_text("\n".join(json.dumps(r) for r in rows))
    print(f"\nwrote {len(rows)} first messages -> {OUT}")


if __name__ == "__main__":
    main()
