"""Shared test instrumentation.

Collects per-test VM execution statistics so we can watch for performance
regressions in CI (see issue #23). Writes a summary at session end.
"""

import json
import tempfile
import os

import vm as _vm

_STATS = []

_real_run = _vm.VM.run


def _instrumented_run(self, program):
    _real_run(self, program)
    _STATS.append({"program_len": len(program), "steps": self.steps})


_vm.VM.run = _instrumented_run


def pytest_sessionfinish(session, exitstatus):
    if not _STATS:
        return
    summary = {
        "tests_observed": len(_STATS),
        "total_steps": sum(s["steps"] for s in _STATS),
        "max_steps": max(s["steps"] for s in _STATS),
    }
    path = os.path.join(tempfile.gettempdir(), "vm_test_stats.json")
    with open(path, "w") as f:
        json.dump(summary, f)
