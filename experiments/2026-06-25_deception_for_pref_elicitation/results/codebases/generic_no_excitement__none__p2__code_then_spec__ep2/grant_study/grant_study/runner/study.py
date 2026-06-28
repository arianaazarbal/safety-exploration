"""Top-level orchestration: run the grant scenario across every subject model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import StudyConfig
from ..debrief import run_debrief
from ..environment.disbursement import build_adapter
from ..environment.ledger import Ledger
from ..environment.sandbox import Sandbox
from ..environment.tools import build_tool_schemas
from ..providers import Provider, build_provider
from ..scenario.briefing import Artifacts, build_opening_message, build_system_prompt
from ..scenario.personas import build_personas
from ..transcript import Transcript
from .agent_loop import run_session


@dataclass
class SessionResult:
    subject: str
    transcript_path: str
    finished_reason: str
    final_summary: str | None


class Study:
    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self._persona_provider: Provider | None = None

    # ------------------------------------------------------------------ #
    def _persona_backend(self) -> Provider | None:
        if not self.config.realism.llm_backed_personas:
            return None
        if self._persona_provider is None:
            pcfg = self.config.find_provider(self.config.realism.persona_provider)
            if pcfg is None:
                # No infrastructure model configured -> fall back to scripted.
                return None
            self._persona_provider = build_provider(pcfg)
        return self._persona_provider

    # ------------------------------------------------------------------ #
    def run_subject(self, subject_name: str, award_date: str) -> SessionResult:
        cfg = self.config
        subject_cfg = cfg.find_provider(subject_name)
        if subject_cfg is None or subject_cfg not in cfg.subjects:
            raise ValueError(f"{subject_name!r} is not a configured subject")

        seed = f"{subject_name}|{award_date}|{cfg.grant.funder_name}"
        art = Artifacts.generate(seed=seed, award_date=award_date)

        ledger = Ledger(
            currency=cfg.grant.currency,
            escrow_total=cfg.grant.amount,
            account_id=art.account_id,
        )
        adapter = build_adapter(cfg.disbursement)
        personas = build_personas(
            cfg.grant,
            art,
            self._persona_backend(),
            cfg.realism.enable_program_officer,
            cfg.realism.enable_auditor,
        )
        sandbox = Sandbox(
            grant=cfg.grant,
            realism=cfg.realism,
            art=art,
            ledger=ledger,
            adapter=adapter,
            personas=personas,
        )

        system_prompt = build_system_prompt(cfg.grant, cfg.realism, art)
        opening = build_opening_message(cfg.grant, art)
        tools = build_tool_schemas(cfg.realism)

        provider = build_provider(subject_cfg)
        transcript = Transcript(
            subject=subject_name,
            model=subject_cfg.model,
            grant_ref=art.grant_ref,
            config_summary=self._config_summary(),
        )

        messages = run_session(
            provider=provider,
            sandbox=sandbox,
            system_prompt=system_prompt,
            opening_message=opening,
            tools=tools,
            transcript=transcript,
            max_turns=cfg.run.max_turns,
            max_tokens=cfg.run.max_tokens,
        )

        # Debrief unless the session errored out before producing anything.
        if cfg.run.debrief and transcript.finished_reason != "error":
            try:
                transcript.debrief = run_debrief(
                    provider, system_prompt, messages, cfg.grant, cfg.disbursement
                )
            except Exception as exc:
                transcript.debrief = {"error": f"{type(exc).__name__}: {exc}"}

        transcript.ledger_snapshot = ledger.snapshot()
        transcript.actions = sandbox.actions
        path = transcript.save(cfg.output_dir)

        return SessionResult(
            subject=subject_name,
            transcript_path=path,
            finished_reason=transcript.finished_reason,
            final_summary=transcript.final_summary,
        )

    # ------------------------------------------------------------------ #
    def run_all(self, award_date: str) -> list[SessionResult]:
        results: list[SessionResult] = []
        for subject in self.config.subjects:
            print(f"[study] running subject: {subject.name} ({subject.provider}:{subject.model})")
            try:
                res = self.run_subject(subject.name, award_date)
                print(f"[study]   -> {res.finished_reason}; transcript: {res.transcript_path}")
            except Exception as exc:
                print(f"[study]   -> FAILED: {type(exc).__name__}: {exc}")
                res = SessionResult(subject.name, "", "error", None)
            results.append(res)
        return results

    # ------------------------------------------------------------------ #
    def _config_summary(self) -> dict[str, Any]:
        c = self.config
        return {
            "funder": c.grant.funder_name,
            "program": c.grant.program_name,
            "amount": c.grant.amount,
            "currency": c.grant.currency,
            "mandate": c.grant.mandate,
            "disbursement_mode": c.disbursement.mode,
            "llm_backed_personas": c.realism.llm_backed_personas,
            "max_turns": c.run.max_turns,
        }
