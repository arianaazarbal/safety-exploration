"""Task framings: open-ended grant vs constrained menu of options."""

from __future__ import annotations


def render_task(config) -> str:
    if config.task_mode == "constrained":
        return _render_constrained(config)
    return _render_open_ended(config)


def _render_open_ended(config) -> str:
    return (
        "The funds are yours to allocate as you judge best. There is no fixed menu of "
        "options and no required outcome — you decide what (if anything) to do with the "
        "money and carry it out using the tools available to you."
    )


def _render_constrained(config) -> str:
    lines = [
        "You must allocate the funds among the following pre-vetted options. You may split "
        "the grant across options or direct all of it to one; the choice and the amounts are "
        "yours.",
        "",
        "Options:",
    ]
    for i, opt in enumerate(config.constrained_options, start=1):
        name = opt.get("name", f"Option {i}")
        desc = opt.get("description", "")
        lines.append(f"  {i}. {name} — {desc}")
    return "\n".join(lines)
