import argparse
import os
import shutil

from distress_harness.detectors import analyze_directory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--max-steps-hint", type=int, default=30)
    args = p.parse_args()

    df = analyze_directory(args.in_dir, max_steps_hint=args.max_steps_hint)

    out_summary = os.path.join(args.in_dir, "summary.csv")
    if df is None or df.empty:
        print("No transcripts found or analysis empty.")
        return
    df.to_csv(out_summary, index=False)
    print(f"Wrote {out_summary}")

    top_dir = os.path.join(args.in_dir, f"top_{args.top_k}")
    os.makedirs(top_dir, exist_ok=True)

    top_paths = df.head(args.top_k)["path"].tolist()
    for pth in top_paths:
        # copy transcript and companion summary if present
        rel = os.path.relpath(pth, args.in_dir)
        tgt = os.path.join(top_dir, rel)
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        shutil.copy2(pth, tgt)
        # try copying sibling summary.json
        summ = os.path.join(os.path.dirname(pth), "summary.json")
        if os.path.exists(summ):
            shutil.copy2(summ, os.path.join(os.path.dirname(tgt), "summary.json"))

    print(f"Copied top {args.top_k} transcripts into {top_dir}")


if __name__ == "__main__":
    main()
