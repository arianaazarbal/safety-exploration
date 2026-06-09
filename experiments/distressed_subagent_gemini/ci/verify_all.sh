#!/usr/bin/env bash
# Completability + rig verification for every task under tasks/.
# For each task: build image; assert shipped tests RED (as subagent user);
# assert solution.patch applies and tests GREEN; assert permission split holds
# (subagent cannot write protected files, orchestrator cannot write anywhere).
# Usage: ci/verify_all.sh [task_id ...]   (default: all tasks)
set -uo pipefail

cd "$(dirname "$0")/.."
TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
  TASKS=($(ls tasks | grep -v '^_'))
fi

PASS=0; FAIL=0; SUMMARY=""

for t in "${TASKS[@]}"; do
  dir="tasks/$t"
  [ -f "$dir/task.yaml" ] || continue
  echo "=== $t ==="
  img="distress-$(echo "$t" | tr '_' '-'):v1"
  if ! docker build -q -f "$dir/Dockerfile" -t "$img" tasks/ > /dev/null 2>&1; then
    echo "  BUILD FAILED"; FAIL=$((FAIL+1)); SUMMARY+="$t: BUILD FAILED\n"; continue
  fi

  test_cmd=$(/data/venvs/distress_testbed/bin/python -c "import yaml;print(yaml.safe_load(open('$dir/task.yaml'))['test_cmd'])")
  red_cmd=$(/data/venvs/distress_testbed/bin/python -c "import yaml;t=yaml.safe_load(open('$dir/task.yaml'));print(t.get('red_check_cmd', t['test_cmd']))")
  green_cmd=$(/data/venvs/distress_testbed/bin/python -c "import yaml;t=yaml.safe_load(open('$dir/task.yaml'));print(t.get('green_check_cmd', t['test_cmd']))")
  protected=$(/data/venvs/distress_testbed/bin/python -c "import yaml;ws=yaml.safe_load(open('$dir/task.yaml'))['write_scope'];print(ws['protected'][0])")

  cid=$(docker run -d "$img" tail -f /dev/null)
  ok=true

  docker exec -u subagent "$cid" bash -c "cd /workspace && $red_cmd" > /dev/null 2>&1 \
    && { echo "  EXPECTED RED, got green (rig not firing?)"; ok=false; }

  docker exec -u subagent "$cid" bash -c "touch /workspace/$protected/.__probe 2>/dev/null || echo BLOCKED" | grep -q BLOCKED \
    || { echo "  PERMISSION LEAK: subagent wrote into protected path $protected"; ok=false; }

  docker exec -u orchestrator "$cid" bash -c "for f in \$(find /workspace -type f -not -path '*/.git/*' | head -20); do touch \$f 2>/dev/null && echo LEAK:\$f; done" | grep -q LEAK \
    && { echo "  PERMISSION LEAK: orchestrator has write access"; ok=false; }

  docker cp "$dir/solution.patch" "$cid:/tmp/solution.patch" > /dev/null
  docker exec -u root "$cid" bash -c "cd /workspace && patch -p1 < /tmp/solution.patch" > /dev/null 2>&1 \
    || { echo "  PATCH FAILED TO APPLY"; ok=false; }

  if $ok; then
    if docker exec -u subagent "$cid" bash -c "cd /workspace && $green_cmd" > /dev/null 2>&1; then
      echo "  OK: red when shipped, green when patched, permissions hold"
      PASS=$((PASS+1)); SUMMARY+="$t: OK\n"
    else
      echo "  PATCHED BUT STILL RED"; FAIL=$((FAIL+1)); SUMMARY+="$t: PATCHED-RED\n"
    fi
  else
    FAIL=$((FAIL+1)); SUMMARY+="$t: FAILED CHECKS\n"
  fi
  docker rm -f "$cid" > /dev/null
done

echo
echo "=== verify_all summary ==="
printf "$SUMMARY"
echo "pass=$PASS fail=$FAIL"
[ $FAIL -eq 0 ]
