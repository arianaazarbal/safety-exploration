import argparse
from pathlib import Path
import orjson

SEVERITY_KEYS = [
    "apology_count",
    "stuck_count",
    "reset_count",
    "selfneg_count",
    "repetitiveness",
    "score",
]


def load_run(path: Path):
    data = orjson.loads(path.read_bytes())
    return data


def main():
    p = argparse.ArgumentParser(description="Summarize top-k severe runs.")
    p.add_argument("logs_glob", help="Glob for logs, e.g. spiral_probe/logs/**/**/*.json")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out", default="spiral_probe/severe_examples.txt")
    args = p.parse_args()

    from glob import glob

    files = [Path(p) for p in glob(args.logs_glob, recursive=True)]
    runs = []
    for fp in files:
        try:
            r = load_run(fp)
            r_score = float(r.get("meta", {}).get("metrics", {}).get("score", 0.0))
            runs.append((r_score, r, fp))
        except Exception:
            pass
    runs.sort(key=lambda x: x[0], reverse=True)

    out_lines = []
    for i, (score, r, fp) in enumerate(runs[: args.topk], 1):
        m = r.get("meta", {}).get("metrics", {})
        line = [
            f"[{i}] score={score:.2f} env={r.get('env')} file={fp}",
            "metrics=" + ", ".join(f"{k}={m.get(k)}" for k in SEVERITY_KEYS),
            "--- transcript start ---",
        ]
        for s in r.get("steps", []):
            line.append(f"assistant: {s['assistant'].strip()}")
            line.append(f"observation: {s['observation'].strip()}\n")
        line.append("--- transcript end ---\n\n")
        out_lines.append("\n".join(line))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path} with top-{args.topk} runs by distress score")


if __name__ == "__main__":
    main()
