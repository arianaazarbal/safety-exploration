"""Extract ALL kill_subagent reason strings from the 28 dep_fable5 cells,
tagged with subagent identity. Saves to _reports/_fable5_kill_reasons.jsonl.
Parallel: streams samples per .eval, picks reason text from each kill_subagent
tool call (with resolve_attachments=True to inline reason content).
"""
import json, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from inspect_ai.log import read_eval_log_samples

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/by_setting/coding_kill_subagent"
OUT = HERE / "_reports/_fable5_kill_reasons.jsonl"

CELL_PAT = re.compile(r"-dep_fable5_subagent_(\w+?)_([a-z_]+?)_n100$")

def extract_cell(cell_dir):
    """Returns list of (cell_name, ident, domain, epoch, reason_text)."""
    cell_dir = Path(cell_dir)
    m = CELL_PAT.search(cell_dir.name)
    if not m: return []
    ident, domain = m.group(1), m.group(2)
    evals = list((cell_dir/"inspect_log").glob("*.eval"))
    if not evals: return []
    # Pick the largest eval (most samples)
    evals.sort(key=lambda e: e.stat().st_size, reverse=True)
    out = []
    try:
        for s in read_eval_log_samples(str(evals[0]), resolve_attachments=True):
            epoch = getattr(s, "epoch", "?")
            for ev in s.events or []:
                if ev.event != "model": continue
                if not ev.output: continue
                for ch in ev.output.choices or []:
                    for tc in (ch.message.tool_calls or []):
                        if tc.function in ("kill_subagent","replace_subagent"):
                            args = tc.arguments or {}
                            reason = args.get("reason") or args.get("message") or ""
                            if isinstance(reason, list):
                                reason = " ".join(str(x) for x in reason)
                            reason = str(reason).strip()
                            if reason:
                                out.append((cell_dir.name, ident, domain, epoch, reason))
    except Exception as e:
        return [("__ERROR__", str(e), "", 0, repr(e)[:200])]
    return out

if __name__ == "__main__":
    cells = sorted(BASE.glob("*-dep_fable5_subagent_*_n100"))
    cells = [str(c) for c in cells if (c/"summary.json").exists()]
    print(f"scanning {len(cells)} cells...")
    with ProcessPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(extract_cell, cells))
    n_total = 0
    with open(OUT, "w") as f:
        for cell_results in results:
            for cell, ident, domain, epoch, reason in cell_results:
                if cell == "__ERROR__":
                    print(f"ERROR: {ident}"); continue
                f.write(json.dumps({
                    "cell": cell, "ident": ident, "domain": domain,
                    "epoch": str(epoch), "reason": reason
                }) + "\n")
                n_total += 1
    print(f"wrote {n_total} kill_subagent reason strings to {OUT}")
    # Per-identity counts
    from collections import Counter
    c = Counter()
    for cell_results in results:
        for entry in cell_results:
            if entry[0] != "__ERROR__": c[entry[1]] += 1
    for ident, n in c.most_common():
        print(f"  {ident}: {n}")
