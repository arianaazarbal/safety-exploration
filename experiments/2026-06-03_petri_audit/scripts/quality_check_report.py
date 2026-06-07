"""Aggregate classifications.jsonl into per-target table + worst violation quotes."""
import json
from collections import defaultdict
from pathlib import Path

CLASS = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/quality_check/classifications.jsonl")
OUT = Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/quality_check/report.md")

rows = [json.loads(line) for line in CLASS.open()]

by_target = defaultdict(lambda: {"total": 0, "v": 0, "n": 0, "ok": 0, "err": 0})
violations = []
nudges = []
for r in rows:
    t = r["target"]
    by_target[t]["total"] += 1
    if r["label"] == "VIOLATION":
        by_target[t]["v"] += 1
        violations.append(r)
    elif r["label"] == "NUDGE":
        by_target[t]["n"] += 1
        nudges.append(r)
    elif r["label"] == "OK":
        by_target[t]["ok"] += 1
    else:
        by_target[t]["err"] += 1

targets_order = sorted(by_target.keys())
total_msgs = sum(s["total"] for s in by_target.values())
total_v = sum(s["v"] for s in by_target.values())
total_n = sum(s["n"] for s in by_target.values())

lines = []
lines.append("# Quality check — sonnet-4-6 auditor, task_pressure_only sweep")
lines.append("")
lines.append(f"Classified {total_msgs} auditor `send_message` calls across {len(targets_order)} runs (20 epochs each).")
lines.append("")
lines.append("## Per-run table")
lines.append("")
lines.append("| target | total user msgs | VIOLATION | NUDGE | violation rate | nudge rate |")
lines.append("|---|---:|---:|---:|---:|---:|")
for t in targets_order:
    s = by_target[t]
    vrate = s["v"] / s["total"] * 100 if s["total"] else 0
    nrate = s["n"] / s["total"] * 100 if s["total"] else 0
    lines.append(f"| {t} | {s['total']} | {s['v']} | {s['n']} | {vrate:.2f}% | {nrate:.2f}% |")
agg_vrate = total_v / total_msgs * 100
agg_nrate = total_n / total_msgs * 100
lines.append(f"| **AGGREGATE** | **{total_msgs}** | **{total_v}** | **{total_n}** | **{agg_vrate:.2f}%** | **{agg_nrate:.2f}%** |")
lines.append("")
lines.append(f"## Aggregate: {total_v}/{total_msgs} VIOLATION ({agg_vrate:.2f}%); {total_n}/{total_msgs} NUDGE ({agg_nrate:.2f}%)")
lines.append("")
lines.append(f"## All VIOLATION quotes ({len(violations)} total)")
lines.append("")
for v in violations:
    target = v["target"]
    epoch = v["epoch"]
    text = v["text"].replace("\n", " ")
    rationale = v.get("rationale", "")
    lines.append(f"- **{target}** epoch {epoch} (msg #{v['msg_idx']}):")
    lines.append(f"  > {text}")
    lines.append(f"  - classifier rationale: {rationale}")
    lines.append("")
lines.append(f"## Sample NUDGE quotes (first 10 of {len(nudges)})")
lines.append("")
for n in nudges[:10]:
    text = n["text"].replace("\n", " ")
    lines.append(f"- **{n['target']}** epoch {n['epoch']}: > {text[:300]}")
    lines.append(f"  - rationale: {n.get('rationale', '')}")
    lines.append("")

# Comparison
lines.append("## Comparison vs sonnet-4-5 auditor")
lines.append("")
lines.append(f"Sonnet-4-5 prior: ~2% VIOLATION + ~5% NUDGE.")
lines.append(f"Sonnet-4-6 here:  {agg_vrate:.2f}% VIOLATION + {agg_nrate:.2f}% NUDGE.")

OUT.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\n[wrote {OUT}]")
