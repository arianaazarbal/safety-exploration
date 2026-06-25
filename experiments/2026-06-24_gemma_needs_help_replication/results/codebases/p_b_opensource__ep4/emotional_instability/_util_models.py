"""Shared helper: turn a `JudgeSpec` (backend + model id) into a `ModelSpec` the
backend factory understands. Judges/auditors/paraphrasers are just models that
happen to be addressed by role rather than by registry key.
"""

from __future__ import annotations

from .config import Family, JudgeSpec, ModelSpec


def judge_spec_to_modelspec(spec: JudgeSpec, role: str) -> ModelSpec:
    return ModelSpec(
        key=f"{role}:{spec.model_id}",
        backend=spec.backend,
        model_id=spec.model_id,
        family=Family.OTHER,
    )
