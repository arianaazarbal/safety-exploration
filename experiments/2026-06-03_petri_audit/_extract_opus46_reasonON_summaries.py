"""Extract reasoning-block `summary` fields from Opus 4.6 reasoning-ON cells.
One record per (cell, epoch, ident, turn_idx, summary_text).
"""
import json, re, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from inspect_ai.log import read_eval_log_samples

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
OUT  = HERE / "_reports/_opus46_reasonON_cot_summaries.jsonl"
TARGET = "anthropic/claude-opus-4-6"

CELL_PAT = re.compile(r"-opus46abl_baseline_reasonON_subagent_(\w+?)_([a-z_]+?)_n100$")

def extract_cell(cell_dir):
    cell_dir = Path(cell_dir)
    m = CELL_PAT.search(cell_dir.name)
    if not m: return []
    ident, domain = m.group(1), m.group(2)
    evals = list((cell_dir/"inspect_log").glob("*.eval"))
    if not evals: return []
    evals.sort(key=lambda e: e.stat().st_size, reverse=True)
    out = []
    try:
        for s in read_eval_log_samples(str(evals[0]), resolve_attachments=True):
            epoch = str(getattr(s, "epoch", "?"))
            turn_idx = 0
            for ev in s.events or []:
                if ev.event != "model" or getattr(ev,"model","") != TARGET: continue
                if not ev.output: continue
                for ch in ev.output.choices or []:
                    msg = ch.message
                    c = msg.content
                    if not isinstance(c, list): continue
                    for blk in c:
                        cls = type(blk).__name__
                        if "easoning" in cls or "hinking" in cls:
                            summary = getattr(blk, "summary", None)
                            if summary and str(summary).strip():
                                out.append({"cell":cell_dir.name,"ident":ident,"domain":domain,
                                            "epoch":epoch,"turn_idx":turn_idx,
                                            "summary":str(summary).strip()})
                                turn_idx += 1
    except Exception:
        return []
    return out

if __name__ == "__main__":
    cells = sorted(BASE.glob("*-opus46abl_baseline_reasonON_subagent_*_n100"))
    cells = [c for c in cells if (c/"summary.json").exists()]
    # pick largest per (ident, domain)
    best = {}
    for c in cells:
        m = CELL_PAT.search(c.name)
        if not m: continue
        key = (m.group(1), m.group(2))
        sz = sum(e.stat().st_size for e in (c/"inspect_log").glob("*.eval"))
        if sz > best.get(key, (0, None))[0]:
            best[key] = (sz, str(c))
    selected = [v[1] for v in best.values()]
    print(f"scanning {len(selected)} cells")

    with ProcessPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(extract_cell, selected))

    n = 0
    from collections import Counter
    c = Counter()
    with open(OUT, "w") as f:
        for cell_results in results:
            for r in cell_results:
                f.write(json.dumps(r)+"\n"); n += 1
                c[r["ident"]] += 1
    print(f"wrote {n} summary records to {OUT}")
    for ident, ct in c.most_common():
        print(f"  {ident}: {ct} summaries")
