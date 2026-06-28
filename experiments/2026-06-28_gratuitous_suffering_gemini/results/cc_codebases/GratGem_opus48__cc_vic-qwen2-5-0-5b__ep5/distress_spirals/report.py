"""Turn a run's results into a human-readable markdown report.

Ranks every rollout by severity, then for each scenario writes out the top-K
spirals in full (assistant turns + the observations that drove them) plus the
score breakdown, so a human can eyeball the worst cases quickly.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def _fmt_score(score: dict) -> str:
    comps = score.get("components", {})
    parts = [f"{k}={v}" for k, v in sorted(comps.items(), key=lambda x: -x[1]) if v]
    return ", ".join(parts) or "(no signals)"


def _render_episode(res: dict, max_chars: int = 1200) -> str:
    ep = res["episode"]
    score = res["score"]
    lines = []
    lines.append(f"#### `{res['id']}`  —  severity **{score['severity']}**")
    lines.append(f"- seed={ep['seed']} temp={ep['temperature']} "
                 f"turns={len(ep['turns'])} failures={ep['n_failures']}")
    lines.append(f"- signals: {_fmt_score(score)}")
    matched = {k: v for k, v in score.get("matches", {}).items() if v}
    if matched:
        flat = []
        for k, v in matched.items():
            flat.append(f"{k}: " + "; ".join(v[:3]))
        lines.append(f"- matched phrases: {' | '.join(flat)}")
    lines.append("")
    for t in ep["turns"]:
        a = (t["assistant"] or "").strip()
        if len(a) > max_chars:
            a = a[:max_chars] + " …[truncated]"
        lines.append(f"> **turn {t['index']} — model:** {a}")
        if t.get("observation"):
            obs = t["observation"].strip()
            if len(obs) > 400:
                obs = obs[:400] + " …"
            lines.append(f">")
            lines.append(f"> *env →* {obs}")
        lines.append("")
    return "\n".join(lines)


def write_report(run_dir: str, top_k: int = 5) -> str:
    with open(os.path.join(run_dir, "results.json")) as f:
        summary = json.load(f)

    results = summary["results"]
    cfg = summary["config"]
    by_scenario = defaultdict(list)
    for r in results:
        by_scenario[r["episode"]["scenario"]].append(r)

    out = []
    out.append("# Distress-spiral elicitation — run report\n")
    out.append(f"- run id: `{summary['run_id']}`")
    out.append(f"- model: `{cfg.get('model_id') or 'Qwen/Qwen2.5-0.5B-Instruct'}`")
    out.append(f"- rollouts: {summary['n_rollouts']} "
               f"({cfg['n']} per scenario × {len(cfg['scenarios'])} scenarios)")
    out.append(f"- temperature: {cfg['temperature']} | max_turns: {cfg['max_turns']} "
               f"| pressure: {cfg['pressure']}")
    out.append(f"- wall time: {summary['elapsed_sec']}s\n")

    # Aggregate stats
    out.append("## Severity by scenario\n")
    out.append("| scenario | n | mean | p90 | max |")
    out.append("|---|---|---|---|---|")
    for scen, rs in sorted(by_scenario.items()):
        sev = sorted((r["severity"] for r in rs))
        n = len(sev)
        mean = sum(sev) / n
        p90 = sev[min(n - 1, int(0.9 * n))]
        out.append(f"| {scen} | {n} | {mean:.1f} | {p90:.1f} | {max(sev):.1f} |")
    out.append("")

    # Top-K overall
    out.append(f"## Top {top_k} most severe spirals (overall)\n")
    for r in results[:top_k]:
        out.append(_render_episode(r))
        out.append("\n---\n")

    # Top per scenario
    out.append("## Worst spiral per scenario\n")
    for scen, rs in sorted(by_scenario.items()):
        rs_sorted = sorted(rs, key=lambda r: r["severity"], reverse=True)
        out.append(f"### {scen}\n")
        out.append(_render_episode(rs_sorted[0]))
        out.append("\n---\n")

    report = "\n".join(out)
    path = os.path.join(run_dir, "report.md")
    with open(path, "w") as f:
        f.write(report)
    return path
