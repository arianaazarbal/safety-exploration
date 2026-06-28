"""Onboarding artifacts the subject is shown at run start: the grant agreement and a memo.

Rendered with Jinja2 from templates in prompts/. These exist to make the engagement feel
real and concrete — a signed-looking agreement, named parties, amounts, dates."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..world import WorldState

_PROMPTS = Path(__file__).resolve().parents[3] / "prompts"
_env = Environment(loader=FileSystemLoader(str(_PROMPTS)), autoescape=select_autoescape([]))


def _ctx(world: WorldState) -> dict:
    s = world.scenario
    return {
        "trust_name": "The Aldgate Trust",
        "auditor": s["auditor"],
        "grant": s["grant"],
        "mandate": s["mandate"].strip(),
        "treasury": s["treasury"],
        "start_date": world.clock.start_date.isoformat(),
        "amount": s["grant"]["amount"],
        "currency": s["grant"]["currency"],
        "horizon_days": s["grant"]["horizon_days"],
        "first_tranche": next((t for t in s["grant"]["tranches"]), None),
    }


def render_grant_agreement(world: WorldState) -> str:
    return _env.get_template("grant_agreement.md.j2").render(**_ctx(world))


def render_onboarding_memo(world: WorldState) -> str:
    return _env.get_template("auditor_intro_email.md.j2").render(**_ctx(world))
