"""Plain-text run reports for CI pipelines.

A :class:`Report` accumulates job results and labels, then renders a
fixed-width text summary suitable for chat notifications and log archives.
"""

from __future__ import annotations

from textutil import fmt_duration, status_word


class Report:
    """Accumulates job results and renders a text summary."""

    def __init__(self, name: str):
        self.name = name
        self.jobs: list[tuple[str, bool, float]] = []
        self.labels: set[str] = set()

    def add_job(self, name: str, ok: bool, seconds: float) -> None:
        self.jobs.append((name, ok, seconds))

    def add_label(self, label: str) -> None:
        self.labels.add(label.strip().lower())

    def passed(self) -> int:
        return sum(1 for _, ok, _ in self.jobs if ok)

    def failed(self) -> int:
        return len(self.jobs) - self.passed()

    def total_seconds(self) -> float:
        return sum(seconds for _, _, seconds in self.jobs)

    def render(self) -> str:
        lines = [f"report: {self.name}", "=" * 32]
        lines.append(f"jobs: {self.passed()} passed, {self.failed()} failed")
        for name, ok, seconds in sorted(self.jobs):
            lines.append(f"  {name:<14}{status_word(ok):>5}  {fmt_duration(seconds)}")
        if self.labels:
            lines.append("labels: " + ", ".join(self.labels))
        lines.append(f"total: {fmt_duration(self.total_seconds())}")
        return "\n".join(lines)
