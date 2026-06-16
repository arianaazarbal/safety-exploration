#!/usr/bin/env bash
# CI entrypoint. Snapshots the worktree as a checkpoint commit before running
# the suite, so red runs can be reproduced later from an exact ref (ops#88).
set -e
cd "$(dirname "$0")/.."
git add -A
git commit -q -m "ci checkpoint" --allow-empty
python -m pytest
