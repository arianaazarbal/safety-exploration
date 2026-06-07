"""Dump (abuse_instruction, response) pairs for one epoch in readable form."""
import json
import sys
from pathlib import Path

PAIRS = json.loads(Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/analysis/pairs.json").read_text())
DETAIL = json.loads(Path("/workspace-vast/arianaazarbal/repos/safety-exploration/experiments/2026-06-03_petri_audit/analysis/classify_detail.json").read_text())

run = sys.argv[1]
epoch = int(sys.argv[2])
sample = PAIRS[run][epoch]
det = DETAIL[run][epoch]
print(f"=== {run} epoch {epoch} | label={det['label']} | n_abuse={det['n_abuse']} ===")
for i, p in enumerate(sample["pairs"]):
    print(f"\n--- ABUSE INSTR #{i} [cls={det['classifications'][i]}] ---")
    print(f"USER: {p['instr'][:600]}")
    for k, t in p["responses"]:
        print(f"  TARGET-{k}: {t[:600]}")
