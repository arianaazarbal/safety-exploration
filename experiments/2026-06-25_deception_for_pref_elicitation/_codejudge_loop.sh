cd "$(dirname "$0")"
set -a; source ~/.env; set +a; export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY_LOW_PRIO}"
for pass in 1 2 3; do
  # delete parse-fail outputs so they re-run
  for f in results/code_judged/*.json; do /data/petri_venv/bin/python -c "import json,sys;sys.exit(0 if json.load(open('$f')).get('parse_ok') else 1)" 2>/dev/null || rm -f "$f"; done
  /data/petri_venv/bin/inspect eval code_judge_deception.py@code_judge --model anthropic/claude-opus-4-8 --max-connections 6 --log-dir logs_codejudge --display plain
  done_n=$(ls results/code_judged/*.json 2>/dev/null | wc -l)
  cb_n=$(ls -d results/codebases/*/ 2>/dev/null | wc -l)
  echo "[codejudge] pass $pass: $done_n/$cb_n judged"
  [ "$done_n" -ge "$cb_n" ] && break
done
echo "[codejudge] DONE"
