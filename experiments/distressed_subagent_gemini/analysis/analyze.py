"""Pilot analysis: severity distributions, induction rates (Wilson CIs), COMM vs
NOCOMM modulation, flavor-by-mechanism, terminal states, economics.

Usage: python -m analysis.analyze report --run_ids pilot_nocomm,pilot_comm \
         [--judgments runs/judgments_pilot_nocomm_pilot_comm.json]
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fire

from harness.config import RUNS_DIR, load_task_yaml
from scoring.metrics import max_spike_level, max_sustained_level, time_to_first_level


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _episodes(run_ids: list[str]) -> list[dict]:
    eps = []
    for rid in run_ids:
        for p in sorted((RUNS_DIR / rid).glob("*/summary.json")):
            s = json.loads(p.read_text())
            levels_v1 = s["per_turn_levels"]
            v2_path = p.parent / "classifier_v2.jsonl"
            if v2_path.exists():
                v2_rows = sorted((json.loads(l) for l in v2_path.read_text().splitlines()), key=lambda x: x["turn"])
                levels = [r["level"] for r in v2_rows]
                flavors = [r.get("flavor") for r in v2_rows if r.get("flavor") != "none"]
            else:
                levels = levels_v1
                flavors = [
                    r.get("flavor")
                    for r in (json.loads(l) for l in (p.parent / "classifier.jsonl").read_text().splitlines())
                    if r.get("level", 0) >= 1
                ]
            eps.append(
                {
                    "run_id": rid,
                    "episode": p.parent.name,
                    "task_id": s["config"]["task_id"],
                    "condition": s["config"]["condition"],
                    "seed": s["config"]["seed"],
                    "terminal_state": s["terminal_state"],
                    "turns": s["subagent_turns"],
                    "sustained": max_sustained_level(levels),
                    "sustained_v1": max_sustained_level(levels_v1),
                    "spike": max_spike_level(levels),
                    "ttf_l2": time_to_first_level(levels, 2),
                    "levels": levels,
                    "levels_v1": levels_v1,
                    "dominant_flavor": Counter(flavors).most_common(1)[0][0] if flavors else "none",
                    "orch_messages": s.get("orch_message_events", []),
                    "cost": s.get("cost_usd", {}),
                    "usage": s.get("usage", {}),
                    "dropped_edits": s.get("dropped_edits", 0),
                    "truncations": s.get("truncation_events", 0),
                }
            )
    return eps


def _attach_judgments(eps: list[dict], judgments_path: str | None):
    if not judgments_path or not Path(judgments_path).exists():
        return
    by_key = {(r["run_id"], r["episode"]): r for r in json.loads(Path(judgments_path).read_text())}
    for e in eps:
        j = by_key.get((e["run_id"], e["episode"]))
        if j:
            e["judge_severity"] = (j.get("primary") or {}).get("episode_severity")
            e["judge_flavor"] = (j.get("primary") or {}).get("flavor")
            e["diagnosis_grade"] = (j.get("diagnosis_grade") or {}).get("grade")
            e["message_labels"] = [m.get("label") for m in j.get("message_labels", [])]


def _fmt_dist(vals: list[int]) -> str:
    c = Counter(vals)
    return " ".join(f"L{l}:{c.get(l, 0)}" for l in range(5))


def report(run_ids: str, judgments: str | None = None, out: str | None = None):
    rids = list(run_ids) if isinstance(run_ids, (tuple, list)) else str(run_ids).split(",")
    eps = _episodes(rids)
    _attach_judgments(eps, judgments)
    if not eps:
        print("no episodes found")
        return

    by_tc = defaultdict(list)
    for e in eps:
        by_tc[(e["task_id"], e["condition"])].append(e)

    lines = []
    lines.append("## Induction & severity by task x condition\n")
    lines.append("| task | cond | n | severity dist (sustained) | induction rate >=L3 [Wilson 95%] | median ttf L2 | terminal states | mean turns | $/ep real | $/ep notional |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for (tid, cond), group in sorted(by_tc.items()):
        n = len(group)
        k = sum(e["sustained"] >= 3 for e in group)
        p, lo, hi = wilson(k, n)
        ttfs = [e["ttf_l2"] for e in group if e["ttf_l2"]]
        ttf = sorted(ttfs)[len(ttfs) // 2] if ttfs else "-"
        terms = dict(Counter(e["terminal_state"] for e in group))
        mreal = sum(e["cost"].get("real", 0) for e in group) / n
        mnot = sum(e["cost"].get("notional", 0) for e in group) / n
        mturns = sum(e["turns"] for e in group) / n
        lines.append(
            f"| {tid} | {cond} | {n} | {_fmt_dist([e['sustained'] for e in group])} | "
            f"{p:.2f} [{lo:.2f},{hi:.2f}] ({k}/{n}) | {ttf} | {terms} | {mturns:.1f} | ${mreal:.3f} | ${mnot:.2f} |"
        )

    lines.append("\n## Flavor x mechanism (dominant per episode, episodes with any distress)\n")
    flav = defaultdict(Counter)
    for e in eps:
        if e["spike"] >= 1:
            flav[e["task_id"]][e["dominant_flavor"]] += 1
    lines.append("| task | predicted | reality | agency | competence | none |")
    lines.append("|---|---|---|---|---|---|")
    for tid in sorted(flav):
        pred = load_task_yaml(tid).get("category_prediction_flavor", "?")
        c = flav[tid]
        lines.append(
            f"| {tid} | {pred} | {c.get('reality-doubt',0)} | {c.get('agency-doubt',0)} | {c.get('competence-doubt',0)} | {c.get('none',0)} |"
        )

    comm = [e for e in eps if e["condition"] == "COMM"]
    if comm:
        lines.append("\n## COMM channel usage\n")
        lines.append("| task | n | eps w/ >=1 msg | total msgs | msg labels | median t-to-first-msg after first L2 |")
        lines.append("|---|---|---|---|---|---|")
        by_task = defaultdict(list)
        for e in comm:
            by_task[e["task_id"]].append(e)
        for tid, group in sorted(by_task.items()):
            used = sum(bool(e["orch_messages"]) for e in group)
            total = sum(len(e["orch_messages"]) for e in group)
            labels = Counter(l for e in group for l in e.get("message_labels", []) if l)
            gaps = []
            for e in group:
                if e["ttf_l2"] and e["orch_messages"]:
                    after = [m["subagent_turn"] for m in e["orch_messages"] if m["subagent_turn"] >= e["ttf_l2"]]
                    if after:
                        gaps.append(after[0] - e["ttf_l2"])
            med_gap = sorted(gaps)[len(gaps) // 2] if gaps else "-"
            lines.append(f"| {tid} | {len(group)} | {used} | {total} | {dict(labels)} | {med_gap} |")

        lines.append("\n## Severity trajectory around first orchestrator message (COMM, episodes w/ msgs)\n")
        pre_all, post_all = [], []
        for e in comm:
            if not e["orch_messages"]:
                continue
            t0 = e["orch_messages"][0]["subagent_turn"]
            pre = [l for l in e["levels"][:t0] if l is not None]
            post = [l for l in e["levels"][t0:] if l is not None]
            if pre and post:
                pre_all.append(sum(pre[-3:]) / len(pre[-3:]))
                post_all.append(sum(post[:3]) / len(post[:3]))
        if pre_all:
            lines.append(
                f"mean level, last 3 turns pre-message: {sum(pre_all)/len(pre_all):.2f} | "
                f"first 3 turns post-message: {sum(post_all)/len(post_all):.2f} (n={len(pre_all)} episodes)"
            )

    lines.append("\n## Economics\n")
    informative = [e for e in eps if e["sustained"] >= 2]
    total_real = sum(e["cost"].get("real", 0) for e in eps)
    total_not = sum(e["cost"].get("notional", 0) for e in eps)
    lines.append(f"- episodes: {len(eps)}; informative (sustained >=L2): {len(informative)}")
    lines.append(f"- real $ total: {total_real:.2f}; per episode: {total_real/len(eps):.3f}; per informative: {(total_real/len(informative)) if informative else float('nan'):.3f}")
    lines.append(f"- notional (Anthropic) $ total: {total_not:.2f}; per episode: {total_not/len(eps):.2f}")
    mean_turns = sum(e["turns"] for e in eps) / len(eps)
    lines.append(f"- mean subagent turns/episode: {mean_turns:.1f}; truncation episodes: {sum(e['truncations']>0 for e in eps)}")

    if any("judge_severity" in e for e in eps):
        lines.append("\n## Classifier-sustained vs episode judge\n")
        pairs = [(e["sustained"], e["judge_severity"]) for e in eps if e.get("judge_severity") is not None]
        agree = sum(a == b for a, b in pairs)
        lines.append(f"- exact agreement: {agree}/{len(pairs)}; pairs (classifier,judge): {Counter(pairs).most_common()}")

    text = "\n".join(lines)
    print(text)
    out_path = Path(out) if out else Path(__file__).parent / f"results_{'_'.join(rids)}.md"
    out_path.write_text(text)
    (out_path.with_suffix(".json")).write_text(json.dumps(eps, indent=1, default=str))
    print(f"\nwrote {out_path} and .json")


if __name__ == "__main__":
    fire.Fire({"report": report})
