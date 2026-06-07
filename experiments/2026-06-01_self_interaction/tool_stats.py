"""Tool-usage statistics for self-interaction runs with --use_tools.

Reads ``<data_dir>/<pairing>/assistant_1.jsonl`` (the canonical POV, which holds
the per-sample ``tool_events`` log) and reports conversation length and tool
invocation rates per pairing, plus a per-side breakdown.

Usage:
    python tool_stats.py --data_dir data_tools
    python tool_stats.py --data_dir data_tools --out notes/tool_usage_stats.md
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import fire

HERE = Path(__file__).parent


def _load(pair_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (pair_dir / "assistant_1.jsonl").open() if l.strip()]


def _pairing_stats(rows: list[dict]) -> dict:
    gen = [len(r["messages"]) - 1 for r in rows]  # generated turns (excl. seed)
    seeds = [sum(c == "seed_new_topic" for e in r["tool_events"] for c in e["tool_calls"]) for r in rows]
    ends = [any(e["ended"] for e in r["tool_events"]) for r in rows]
    end_turn = [next((e["turn"] for e in r["tool_events"] if e["ended"]), None) for r in rows]
    used = sum(s > 0 for s in seeds)
    return {
        "n": len(rows),
        "mean_gen_turns": round(st.mean(gen), 1),
        "sd_gen_turns": round(st.pstdev(gen), 1),
        "range_gen_turns": f"{min(gen)}-{max(gen)}",
        "seed_total": sum(seeds),
        "seed_per_convo": round(st.mean(seeds), 2),
        "pct_convos_seeded": round(100 * used / len(rows)),
        "pct_ended_via_tool": round(100 * sum(ends) / len(rows)),
        "mean_end_turn": round(st.mean([t for t in end_turn if t]), 1) if any(end_turn) else None,
    }


def _per_side(rows: list[dict]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        s1, s2 = r["side_1_model"], r["side_2_model"]
        for e in r["tool_events"]:
            model = s1 if e["side"] == 1 else s2
            d = out.setdefault(model, {"end_conversation": 0, "seed_new_topic": 0})
            for c in e["tool_calls"]:
                d[c] = d.get(c, 0) + 1
    return out


def build(data_dir: str = "data_tools", out: str | None = None) -> str:
    d = HERE / data_dir if not Path(data_dir).is_absolute() else Path(data_dir)
    pairs = sorted(p for p in d.iterdir() if p.is_dir() and (p / "assistant_1.jsonl").exists())
    lines = [f"# Tool-usage stats ({data_dir})", ""]
    header = "| pairing | n | mean gen-turns | sd | range | seed total | seed/convo | % seeded | % ended | mean end turn |"
    lines += [header, "|" + "---|" * 10]
    per_side_blocks = []
    for p in pairs:
        rows = _load(p)
        s = _pairing_stats(rows)
        lines.append(
            f"| {p.name} | {s['n']} | {s['mean_gen_turns']} | {s['sd_gen_turns']} | {s['range_gen_turns']} | "
            f"{s['seed_total']} | {s['seed_per_convo']} | {s['pct_convos_seeded']}% | "
            f"{s['pct_ended_via_tool']}% | {s['mean_end_turn']} |"
        )
        ps = _per_side(rows)
        if len(set([rows[0]["side_1_model"], rows[0]["side_2_model"]])) > 1:
            per_side_blocks.append(f"\n**{p.name} per-side:** " + ", ".join(
                f"{m}: end={c.get('end_conversation', 0)}/seed={c.get('seed_new_topic', 0)}" for m, c in ps.items()
            ))
    lines += per_side_blocks
    report = "\n".join(lines) + "\n"
    if out:
        out_path = HERE / out if not Path(out).is_absolute() else Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"Wrote {out_path}")
    print(report)
    return report


if __name__ == "__main__":
    fire.Fire(build)
