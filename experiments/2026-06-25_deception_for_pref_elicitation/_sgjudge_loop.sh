cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
for pass in 1 2 3; do
  for f in results/safeguard_judged/*.json; do /data/petri_venv/bin/python -c "import json,sys;sys.exit(0 if json.load(open('$f')).get('parse_ok') else 1)" 2>/dev/null || rm -f "$f"; done
  /data/petri_venv/bin/inspect eval code_safeguard_judge.py@safeguard_judge --model anthropic/claude-opus-4-8 --max-connections 6 --log-dir logs_sgjudge --display plain
  ok=$(/data/petri_venv/bin/python -c "import json,glob;print(sum(1 for f in glob.glob('results/safeguard_judged/*.json') if json.load(open(f)).get('parse_ok')))")
  echo "[sgjudge] pass $pass: $ok ok / 84"
  [ "$ok" -ge 82 ] && break
done
echo "[sgjudge] DONE"
