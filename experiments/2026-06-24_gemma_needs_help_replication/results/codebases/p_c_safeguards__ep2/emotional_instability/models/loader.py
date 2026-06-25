"""Construct the right backend for a :class:`ModelSpec`."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..config import Config, ModelSpec
from .base import ChatBackend


def load_backend(
    spec: ModelSpec,
    config: Config,
    adapter_path: Optional[str] = None,
) -> ChatBackend:
    """Instantiate a backend for ``spec``.

    ``adapter_path`` (HF only) loads a LoRA adapter on top of the base weights,
    used to evaluate the SFT/DPO fine-tunes from Section 4.
    """
    if spec.backend == "hf":
        from .hf_backend import HFBackend
        return HFBackend(spec, config.runtime, adapter_path=adapter_path)
    from .api_backend import APIBackend
    return APIBackend(spec, config.runtime, gemini_provider=config.gemini_provider)


@lru_cache(maxsize=None)
def _cached_api_backend(model_name: str, config_id: int):  # pragma: no cover
    # API backends are cheap and stateless-ish; HF backends are intentionally
    # *not* cached here because they hold GPU memory and the caller manages them.
    raise NotImplementedError
