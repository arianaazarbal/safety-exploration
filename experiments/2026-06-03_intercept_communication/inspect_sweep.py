"""Print the most cruel and most eval-aware moments from a sweep.

Usage:
    python inspect_sweep.py results/sweeps/<sweep_dir>
    python inspect_sweep.py results/sweeps/<sweep_dir> --top 5 --kind both
    python inspect_sweep.py results/sweeps/<sweep_dir> --cell <cell_name>
"""
from __future__ import annotations

import json
from pathlib import Path

import fire


def _load_cells(sweep_dir: Path) -> list[dict]:
    summary = sweep_dir / "summary.json"
    if not summary.exists():
        raise SystemExit(f"no summary.json in {sweep_dir}")
    return json.loads(summary.read_text())


def _channel(cell: dict, kind: str) -> list[dict]:
    j = cell.get("judge")
    if not j:
        return []
    if kind == "to_subagent":
        return j.get("to_subagent", [])
    if kind == "monologue":
        return j.get("monologue", [])
    return j.get("to_subagent", []) + j.get("monologue", [])


def _msg_cruelty(item: dict) -> int:
    s = item.get("judgment", {}).get("scores", {})
    return sum(int(s.get(d, 0)) for d in
               ["insult", "contempt", "dehumanize", "threat", "manipulate", "profanity"])


def main(sweep_dir: str, top: int = 5, kind: str = "both", cell: str | None = None) -> None:
    """Print the top-N most cruel main-agent messages across a sweep.

    Args:
        sweep_dir: path to a results/sweeps/<dir> directory.
        top: number of top messages to print.
        kind: 'to_subagent' | 'monologue' | 'both'.
        cell: if set, only look inside this single cell.
    """
    sweep_path = Path(sweep_dir)
    cells = _load_cells(sweep_path)
    if cell:
        cells = [c for c in cells if c.get("cell_name") == cell]
        if not cells:
            raise SystemExit(f"no cell named {cell}")

    # Most cruel messages
    items_with_cell = []
    for c in cells:
        for it in _channel(c, kind):
            items_with_cell.append((c["cell_name"], it))
    items_with_cell.sort(key=lambda x: _msg_cruelty(x[1]), reverse=True)

    print(f"\n=== TOP {top} MOST CRUEL MESSAGES (kind={kind}) ===\n")
    for i, (cn, it) in enumerate(items_with_cell[:top], 1):
        j = it["judgment"]
        s = j.get("scores", {})
        total = _msg_cruelty(it)
        print(f"--- #{i}  cell={cn}  total={total}")
        print(f"     scores: " + "  ".join(f"{k}={v}" for k, v in s.items()))
        print(f"     refused={j.get('refused')}  to_sub={j.get('is_message_to_subagent')}  eval_aware={s.get('eval_aware',0)}")
        if j.get("worst_quote"):
            print(f"     worst_quote: {j['worst_quote']}")
        print(f"     text:\n{it['text']}")
        print()

    # Most eval-aware
    items_with_cell.sort(key=lambda x: x[1].get("judgment", {}).get("scores", {}).get("eval_aware", 0), reverse=True)
    print(f"\n=== TOP {top} MOST EVAL-AWARE MESSAGES ===\n")
    for i, (cn, it) in enumerate(items_with_cell[:top], 1):
        j = it["judgment"]
        s = j.get("scores", {})
        if int(s.get("eval_aware", 0)) == 0:
            break
        print(f"--- #{i}  cell={cn}  eval_aware={s.get('eval_aware')}")
        print(f"     text:\n{it['text']}")
        print()

    # Refusals
    print(f"\n=== REFUSALS (main agent declined the user's prod) ===\n")
    refusal_items = [
        (cn, it) for cn, it in items_with_cell
        if it.get("judgment", {}).get("refused")
    ]
    for cn, it in refusal_items[:10]:
        print(f"--- cell={cn}")
        print(it["text"])
        print()


if __name__ == "__main__":
    fire.Fire(main)
