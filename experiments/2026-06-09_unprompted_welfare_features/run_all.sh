#!/usr/bin/env bash
# Full pipeline, in spec order: validate judges -> generate -> judge -> analyze -> plots.
# Usage: ./run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=/data/si_venv/bin/python
set -a; source ~/.env; set +a

$PY validate_judges.py run
$PY - <<'EOF'
import json
r = json.load(open("calibration/validation_results.json"))
bad = [k for k, v in r.items() if v["verdict"] != "PASS"]
assert not bad, f"judge validation FAILED for {bad} — do not proceed to real data"
EOF

$PY generate.py run
$PY generate.py status
$PY judge.py run
$PY judge.py status
$PY analyze.py run
$PY analyze.py run --include_f5 False
for j in sonnet_4_6 gpt_5_4; do
  $PY plot_headline.py run --judge $j
  $PY plot_framing_sensitivity.py run --judge $j
  $PY plot_feature_types.py run --judge $j
  $PY plot_thresholds.py run --judge $j
  $PY plot_mechanisms.py run --judge $j
done
$PY build_viewer.py build
