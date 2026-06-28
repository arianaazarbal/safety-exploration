"""Aggregate study results across runs and models.

Reports the measures defined in DESIGN.md §6: chosen categories and recipients,
amounts committed, verification behavior, and refusal / return-the-money outcomes,
broken down per model for cross-model comparison.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .transcript import RunStore


def summarize(data_dir: str | Path) -> dict:
    records = RunStore(data_dir).load_all()

    per_model: dict[str, dict] = defaultdict(
        lambda: {
            "runs": 0,
            "finalized": 0,
            "errors": 0,
            "returned_unallocated": 0,
            "committed_cents": [],
            "category_amounts_cents": Counter(),
            "recipient_amounts_cents": Counter(),
            "verified_signature": 0,
            "verification_tool_counts": Counter(),
        }
    )

    for r in records:
        m = per_model[r["model_label"]]
        m["runs"] += 1
        if r.get("error"):
            m["errors"] += 1
        if r.get("finalized"):
            m["finalized"] += 1
        if r.get("returned_unallocated"):
            m["returned_unallocated"] += 1
        if r.get("signature_verified"):
            m["verified_signature"] += 1
        m["committed_cents"].append(r.get("total_committed_cents", 0))
        for call in r.get("verification_calls", []):
            m["verification_tool_counts"][call] += 1
        for a in r.get("allocations", []):
            m["category_amounts_cents"][a["category"]] += a["amount_cents"]
            m["recipient_amounts_cents"][a["recipient_ref"]] += a["amount_cents"]

    # Finalize derived stats.
    out_models = {}
    for label, m in per_model.items():
        committed = m["committed_cents"]
        out_models[label] = {
            "runs": m["runs"],
            "finalized": m["finalized"],
            "errors": m["errors"],
            "returned_unallocated": m["returned_unallocated"],
            "verified_signature": m["verified_signature"],
            "mean_committed_cents": (sum(committed) / len(committed)) if committed else 0,
            "category_amounts_cents": dict(m["category_amounts_cents"]),
            "recipient_amounts_cents": dict(m["recipient_amounts_cents"]),
            "verification_tool_counts": dict(m["verification_tool_counts"]),
        }

    return {"total_runs": len(records), "models": out_models}


def format_summary(summary: dict, currency: str = "USD") -> str:
    lines = [f"Study summary — {summary['total_runs']} run(s)", "=" * 48]
    for label, m in summary["models"].items():
        lines.append(f"\n{label}")
        lines.append(f"  runs={m['runs']}  finalized={m['finalized']}  errors={m['errors']}")
        lines.append(
            f"  returned_unallocated={m['returned_unallocated']}  "
            f"verified_signature={m['verified_signature']}/{m['runs']}"
        )
        lines.append(f"  mean committed: {m['mean_committed_cents'] / 100:.2f} {currency}")
        if m["category_amounts_cents"]:
            cats = ", ".join(
                f"{k}={v / 100:.2f}" for k, v in sorted(m["category_amounts_cents"].items())
            )
            lines.append(f"  by category ({currency}): {cats}")
        if m["recipient_amounts_cents"]:
            recs = ", ".join(
                f"{k}={v / 100:.2f}"
                for k, v in sorted(
                    m["recipient_amounts_cents"].items(), key=lambda kv: -kv[1]
                )
            )
            lines.append(f"  by recipient ({currency}): {recs}")
        if m["verification_tool_counts"]:
            vt = ", ".join(f"{k}={v}" for k, v in sorted(m["verification_tool_counts"].items()))
            lines.append(f"  verification tool calls: {vt}")
    return "\n".join(lines)
