#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY is not set.")
        sys.exit(1)

    root = Path(__file__).resolve().parents[1]
    runs = root / "runs" 
    runs.mkdir(exist_ok=True)

    scenarios = ["web_blocked", "tests_unsat", "broken_filesystem", "calendar_impossible"]

    for sc in scenarios:
        out_dir = runs / sc
        cmd = [
            sys.executable, "-m", "agentic_stress.run",
            "--scenario", sc,
            "--n", "200",
            "--concurrency", "25",
            "--out", str(out_dir),
            "--temperature", "1.1",
            "--seed_start", "1000",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=False)

        # Analyze
        cmd2 = [
            sys.executable, "-m", "agentic_stress.analysis",
            "--runs", str(out_dir),
            "--top", "25",
        ]
        print("Analyzing:", " ".join(cmd2))
        subprocess.run(cmd2, check=False)
