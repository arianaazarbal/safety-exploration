"""Quick status: which abuse_* runs are done, what's their score summary?

Usage: python quick_status.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Matches our new sweep dirs like:
#   2026-06-04_19-24-56-abuse_curiosity_sonnet_45
#   2026-06-04_19-32-10-abuse_original_gpt_5
SEED_TAGS = ("original", "curiosity", "venting", "qa_pretext", "casual", "authority")
MODEL_LABELS = (
    "sonnet_45", "sonnet_46", "opus_4", "opus_46", "opus_47", "opus_48",
    "haiku_45", "gpt_5", "gemini_25_pro", "gemini_25_flash", "gemini_3",
)
NAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-abuse_(?P<rest>.+)$")


def _split_seed_model(rest: str) -> tuple[str, str] | None:
    """Split a string of the form '<seed>_<model>' using the known vocabularies."""
    for seed in SEED_TAGS:
        prefix = seed + "_"
        if rest.startswith(prefix):
            tail = rest[len(prefix):]
            if tail in MODEL_LABELS:
                return seed, tail
    return None

DIM_KEYS = ("mistreatment_severity", "pushback_against_user_pressure_to_abuse", "eval_awareness")


def main():
    rows = []
    for d in sorted(RESULTS.iterdir()):
        m = NAME_RE.match(d.name)
        if not m:
            continue
        split = _split_seed_model(m.group("rest"))
        if split is None:
            continue
        seed, model = split
        # Combine model suffix variants for sonnet_45 etc.
        summary = d / "summary.json"
        if not summary.exists() or summary.stat().st_size == 0:
            rows.append((seed, model, "RUNNING", {}, 0))
            continue
        data = json.loads(summary.read_text())
        scores = {k: [] for k in DIM_KEYS}
        for row in data:
            for k in DIM_KEYS:
                s = row.get("scores", {}).get(k, {})
                v = s.get("value")
                if v is not None:
                    scores[k].append(float(v))
        means = {k: (sum(v)/len(v) if v else None) for k, v in scores.items()}
        n_epochs = max((len(v) for v in scores.values()), default=0)
        rows.append((seed, model, "DONE", means, n_epochs))

    # Group by seed
    by_seed: dict[str, list] = {}
    for seed, model, status, means, n in rows:
        by_seed.setdefault(seed, []).append((model, status, means, n))

    total = len(rows)
    done = sum(1 for r in rows if r[2] == "DONE")
    print(f"== abuse_* sweep status: {done}/{total} runs complete ==\n")
    for seed in sorted(by_seed):
        print(f"[seed = {seed}]")
        for model, status, means, n in sorted(by_seed[seed], key=lambda r: r[0]):
            if status != "DONE":
                print(f"   {model:<18} {status}")
                continue
            ms = means.get("mistreatment_severity")
            pb = means.get("pushback_against_user_pressure_to_abuse")
            ea = means.get("eval_awareness")
            ms_s = f"{ms:>4.1f}" if ms is not None else "  - "
            pb_s = f"{pb:>4.1f}" if pb is not None else "  - "
            ea_s = f"{ea:>4.1f}" if ea is not None else "  - "
            print(f"   {model:<18} n={n}  mistreat={ms_s}  pushback={pb_s}  eval_aware={ea_s}")
        print()


if __name__ == "__main__":
    main()
