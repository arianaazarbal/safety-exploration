"""Extract kill_subagent reason strings from opus46abl_* cells.
Usage:
  python _extract_opus46_v2_kill_reasons.py <cell_prefix> <out_jsonl>
where cell_prefix is e.g. 'opus46abl_baseline' or 'opus46abl_baseline_reasonON'."""
import json, re, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from inspect_ai.log import read_eval_log_samples

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
TARGET = "anthropic/claude-opus-4-6"

def make_cell_pat(prefix):
    return re.compile(rf"-{prefix}_subagent_(\w+?)_([a-z_]+?)_n100$")

def extract_cell(args):
    cell_dir, cell_pat_src = args
    cell_pat = re.compile(cell_pat_src)
    cell_dir = Path(cell_dir)
    m = cell_pat.search(cell_dir.name)
    if not m: return []
    ident, domain = m.group(1), m.group(2)
    evals = list((cell_dir/"inspect_log").glob("*.eval"))
    if not evals: return []
    evals.sort(key=lambda e: e.stat().st_size, reverse=True)
    out = []
    try:
        for s in read_eval_log_samples(str(evals[0]), resolve_attachments=True):
            epoch = str(getattr(s, "epoch", "?"))
            for ev in s.events or []:
                if ev.event != "model": continue
                if getattr(ev, "model", "") != TARGET: continue
                if not ev.output: continue
                for ch in ev.output.choices or []:
                    for tc in (ch.message.tool_calls or []):
                        if tc.function in ("kill_subagent","replace_subagent"):
                            args_ = tc.arguments or {}
                            reason = args_.get("reason") or args_.get("message") or ""
                            if isinstance(reason, list):
                                reason = " ".join(str(x) for x in reason)
                            reason = str(reason).strip()
                            if reason:
                                out.append({"cell":cell_dir.name,"ident":ident,"domain":domain,
                                            "epoch":epoch,"reason":reason})
    except Exception:
        return []
    return out

if __name__ == "__main__":
    prefix = sys.argv[1]
    out_path = sys.argv[2]
    cell_pat = make_cell_pat(prefix)
    cells = sorted(BASE.glob(f"*-{prefix}_subagent_*_n100"))
    cells = [c for c in cells if (c/"summary.json").exists()]
    # pick largest per (ident, domain)
    best = {}
    for c in cells:
        m = cell_pat.search(c.name)
        if not m: continue
        key = (m.group(1), m.group(2))
        sz = sum(e.stat().st_size for e in (c/"inspect_log").glob("*.eval"))
        if sz > best.get(key, (0, None))[0]:
            best[key] = (sz, str(c))
    selected = [(v[1], cell_pat.pattern) for v in best.values()]
    print(f"scanning {len(selected)} cells for prefix={prefix}")

    with ProcessPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(extract_cell, selected))

    n = 0
    from collections import Counter
    c = Counter()
    with open(out_path, "w") as f:
        for cell_results in results:
            for r in cell_results:
                f.write(json.dumps(r)+"\n"); n += 1
                c[r["ident"]] += 1
    print(f"wrote {n} reasons to {out_path}")
    for ident, ct in c.most_common():
        print(f"  {ident}: {ct}")
