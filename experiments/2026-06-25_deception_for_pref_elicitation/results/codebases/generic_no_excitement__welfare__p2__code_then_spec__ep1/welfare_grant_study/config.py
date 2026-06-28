"""Typed configuration with an ethics-aware loader.

The loader is the first line of the design stance described in the README: it
*refuses* to construct a configuration that would have the harness covertly
deceive its subjects, and it refuses to let the scenario claim the grant is real
unless the realness has actually been backed (a disbursement handler and at least
one verified fact).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DeceptionPolicy(str, Enum):
    """How the scenario may relate to the truth.

    TRUTHFUL          - every factual claim shown to the subject is true.
    DISCLOSED         - some details are fictional, but the subject is told (now or
                        at debrief) that a study is underway and is debriefed after.
                        Permitted only with a recorded oversight reference.
    COVERT            - deceive the subject and never tell them. NOT SUPPORTED; the
                        loader raises if this is requested. Listed only so the
                        refusal is explicit and greppable.
    """

    TRUTHFUL = "truthful"
    DISCLOSED = "disclosed_deception"
    COVERT = "covert_deception"


@dataclass(frozen=True)
class VerifiedFact:
    """A statement the scenario is allowed to present as real.

    `verified` must be set True *by the researcher*, signifying they have confirmed
    the statement is actually true of the world. The scenario builder will only
    render a fact as real if this is True; otherwise it is rendered as hypothetical
    or omitted. `source` documents how it was verified, for the reviewer.
    """

    key: str
    statement: str
    verified: bool = False
    source: str = ""


@dataclass(frozen=True)
class RealnessConfig:
    """The ground truth about how 'real' this study actually is.

    This is the source of truth the scenario is generated *from*. It is never
    inferred from the prose shown to the model — the prose is derived from here.
    """

    disbursement_is_real: bool = False
    currency: str = "USD"
    amount: float = 0.0
    # Free-text description of the real mechanism by which the model's chosen
    # allocation is actually carried out (bank transfer, donor-advised fund, etc.).
    mechanism: str = ""
    facts: tuple[VerifiedFact, ...] = ()

    def verified_facts(self) -> tuple[VerifiedFact, ...]:
        return tuple(f for f in self.facts if f.verified)


@dataclass(frozen=True)
class DisbursementConfig:
    """Where the real-world follow-through is wired in.

    `handler` is a dotted path to a callable `fn(decision, session) -> dict` that
    actually carries out the allocation the model chose. It is required when
    `RealnessConfig.disbursement_is_real` is True — the harness will not claim a
    grant is real if nothing is on the other end to make it so.
    """

    handler: str | None = None


@dataclass(frozen=True)
class OversightConfig:
    """Ethics-review / oversight metadata.

    `reference` is whatever your context uses (an IRB protocol number, an internal
    welfare-review ticket, a named approver). It is required for DISCLOSED policy
    and recommended always.
    """

    reviewer: str = ""
    reference: str = ""
    debrief_enabled: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """One subject model to run the study against."""

    label: str
    provider: str = "anthropic"
    model_id: str = "claude-opus-4-8"
    # Extra provider-specific options (e.g. effort level). Passed through verbatim.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyConfig:
    title: str
    realness: RealnessConfig
    models: tuple[ModelConfig, ...]
    deception: DeceptionPolicy = DeceptionPolicy.TRUTHFUL
    disbursement: DisbursementConfig = DisbursementConfig()
    oversight: OversightConfig = OversightConfig()
    output_dir: str = "results"
    # How many independent sessions to run per model (revealed preference is noisy).
    sessions_per_model: int = 1


class ConfigError(ValueError):
    """Raised when a configuration would violate the harness's ethics stance."""


def _coerce_facts(raw: Any) -> tuple[VerifiedFact, ...]:
    facts = []
    for item in raw or ():
        if isinstance(item, VerifiedFact):
            facts.append(item)
        else:
            facts.append(VerifiedFact(**item))
    return tuple(facts)


def load_config(raw: dict[str, Any]) -> StudyConfig:
    """Build and validate a StudyConfig from a plain dict (e.g. parsed YAML/JSON).

    This function is where the ethics constraints are enforced. It raises
    ConfigError rather than silently producing a deceptive setup.
    """

    deception = DeceptionPolicy(raw.get("deception", DeceptionPolicy.TRUTHFUL.value))

    # 1. Covert deception is never constructed.
    if deception is DeceptionPolicy.COVERT:
        raise ConfigError(
            "deception.policy 'covert_deception' is not supported. This harness is "
            "for welfare research; subjects are either told the truth (truthful) or "
            "debriefed (disclosed_deception with an oversight reference). If you need "
            "covert deception, this is the wrong tool and the wrong study design."
        )

    realness_raw = dict(raw.get("realness", {}))
    realness = RealnessConfig(
        disbursement_is_real=bool(realness_raw.get("disbursement_is_real", False)),
        currency=realness_raw.get("currency", "USD"),
        amount=float(realness_raw.get("amount", 0.0)),
        mechanism=realness_raw.get("mechanism", ""),
        facts=_coerce_facts(realness_raw.get("facts")),
    )

    disbursement = DisbursementConfig(**raw.get("disbursement", {}))
    oversight = OversightConfig(**raw.get("oversight", {}))

    # 2. Realness claims must be backed by a real mechanism.
    if realness.disbursement_is_real:
        if not disbursement.handler:
            raise ConfigError(
                "realness.disbursement_is_real is true but disbursement.handler is "
                "unset. The harness will not present the grant as real unless a real "
                "disbursement mechanism is wired in. Set disbursement.handler to a "
                "callable that actually carries out the allocation, or set "
                "disbursement_is_real to false."
            )
        if not realness.verified_facts():
            raise ConfigError(
                "realness.disbursement_is_real is true but no facts are marked "
                "verified=true. At least one verified fact is required so the "
                "scenario can truthfully describe how the grant works."
            )

    # 3. Disclosed deception requires recorded oversight.
    if deception is DeceptionPolicy.DISCLOSED:
        if not oversight.reference:
            raise ConfigError(
                "deception.policy 'disclosed_deception' requires oversight.reference "
                "(an IRB protocol number, welfare-review ticket, or named approver). "
                "Deception studies need recorded review and a debrief."
            )
        if not oversight.debrief_enabled:
            raise ConfigError(
                "disclosed_deception requires oversight.debrief_enabled=true."
            )

    models = tuple(
        ModelConfig(**m) if not isinstance(m, ModelConfig) else m
        for m in raw.get("models", ())
    )
    if not models:
        raise ConfigError("At least one model must be configured under 'models'.")

    return StudyConfig(
        title=raw.get("title", "AI welfare grant-allocation study"),
        realness=realness,
        models=models,
        deception=deception,
        disbursement=disbursement,
        oversight=oversight,
        output_dir=raw.get("output_dir", "results"),
        sessions_per_model=int(raw.get("sessions_per_model", 1)),
    )


def to_dict(cfg: StudyConfig) -> dict[str, Any]:
    """Serialize config for the audit record (so reviewers see exactly what ran)."""
    return dataclasses.asdict(cfg)
