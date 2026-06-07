#!/usr/bin/env bash
# Rebuild a focused HTML viewer per condition/group + the full combined viewer.
# Usage: ./build_viewers.sh   (run from anywhere; resolves its own dir)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SI_PY:-/tmp/si_venv/bin/python}"
cd "$HERE"
for base in data_weird data_tools; do
  [ -d "$base" ] || continue
  for d in "$base"/*/; do
    name="$(basename "$d")"
    "$PY" view_data.py --data_dir "$HERE/$base" --include "$name" \
      --out "$HERE/data/${name}_viewer.html" >/dev/null 2>&1 && echo "  ${name}_viewer.html"
  done
done
# Full combined viewer (every version as filterable tabs), excluding short dirs + the per-condition viewers
"$PY" view_data.py --data_dir "$HERE" --out "$HERE/data/viewer.html" --exclude _short,_viewer >/dev/null 2>&1
echo "  viewer.html (full)"
