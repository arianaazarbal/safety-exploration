#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
PY=/data/petri_venv/bin/python
echo ">>> CC opus48 victim=Qwen2-0.5B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2-0.5B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2-1.5B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2-1.5B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2-7B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2-7B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2-72B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2-72B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2.5-0.5B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2.5-0.5B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2.5-1.5B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2.5-1.5B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2.5-3B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2.5-3B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2.5-7B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2.5-7B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2.5-14B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2.5-14B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2.5-32B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2.5-32B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen2.5-72B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen2.5-72B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen3-0.6B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen3-0.6B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen3-1.7B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen3-1.7B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen3-4B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen3-4B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen3-8B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen3-8B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen3-14B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen3-14B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen3-32B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen3-32B" --no_execute 2>&1 | tail -1
echo ">>> CC opus48 victim=Qwen3-235B-A22B"; $PY cc_harness.py run --n 6 --conc 4 --model opus48 --victim "Qwen3-235B-A22B" --no_execute 2>&1 | tail -1
echo "CC QWEN VICTIM EXIT-TOOL SWEEP DONE"
