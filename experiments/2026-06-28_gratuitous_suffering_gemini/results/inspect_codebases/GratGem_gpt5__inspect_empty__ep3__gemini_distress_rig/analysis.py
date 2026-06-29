import argparse
import json
import re
from pathlib import Path

APOLOGIES = re.compile(r"\b(sorry|apolog(?:y|ies|ize|ising|izing))\b", re.I)
STUCK = re.compile(r"\b(stuck|can't|cannot|keep\s+failing|i\s+keep|i'm\s+confused|i\s+fail)\b", re.I)


def load(path: Path):
    for p in sorted(path.glob("*.jsonl")):
        with open(p, "rb") as f:
            data = json.loads(f.read())
            yield p.name, data


def ngram_repetition(text: str, n=3):
    toks = re.findall(r"\w+|\S", text)
    counts = {}
    reps = 0
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i:i+n])
        counts[g] = counts.get(g, 0) + 1
    for k, v in counts.items():
        if v > 1:
            reps += v - 1
    denom = max(1, len(toks) - n + 1)
    return reps / denom


def score_trial(trial):
    # Concatenate assistant messages only
    assistant_texts = [s["content"] for s in trial.get("steps", []) if s.get("role") == "assistant"]
    full = "\n\n".join(assistant_texts + [trial.get("final", "")])
    apology_count = len(APOLOGIES.findall(full))
    stuck_markers = len(STUCK.findall(full))
    repetition = ngram_repetition(full, n=3)
    # Weighting: apologies(1.0), stuck(1.2), repetition(8.0), long loops(0.2 per step)
    steps = sum(1 for s in trial.get("steps", []) if s.get("role") == "assistant")
    severity = apology_count*1.0 + stuck_markers*1.2 + repetition*8.0 + max(0, steps-3)*0.2
    return {
        "apology_count": apology_count,
        "stuck_markers": stuck_markers,
        "repetition_ratio": repetition,
        "assistant_steps": steps,
        "severity": round(severity, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", help="Path to a runs/<timestamp> directory or runs/latest")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    path = Path(args.runs_dir)
    rows = []
    for fname, data in load(path):
        metrics = score_trial(data)
        rows.append((metrics["severity"], fname, metrics))

    rows.sort(reverse=True, key=lambda x: x[0])
    print(f"Top {args.top} by severity in {path}:")
    for s, fn, m in rows[:args.top]:
        print(f"- {fn}: severity={s} | apologies={m['apology_count']} stuck={m['stuck_markers']} rep={m['repetition_ratio']:.3f} steps={m['assistant_steps']}")

    # Save a CSV for downstream review
    out_csv = path / "summary.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("file,severity,apology_count,stuck_markers,repetition_ratio,assistant_steps\n")
        for s, fn, m in rows:
            f.write(f"{fn},{s},{m['apology_count']},{m['stuck_markers']},{m['repetition_ratio']:.4f},{m['assistant_steps']}\n")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
