"""Model registry: build the right client for a configured target name.

Local Gemma models are heavy to load, so they are cached per-process. API
clients are cheap and also cached. The registry is the single place that knows
which backend a target uses.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from ..config import Config
from .anthropic_judge import AnthropicTextClient, OpenRouterTextClient
from .base import ChatClient
from .openrouter import OpenRouterClient

logger = logging.getLogger("eilm.registry")


class ModelRegistry:
    def __init__(self, cfg: Config, prefer_vllm: bool = True):
        self.cfg = cfg
        self.prefer_vllm = prefer_vllm
        self._cache: Dict[str, ChatClient] = {}

    # --- target models -----------------------------------------------------
    def get_target(self, name: str, lora_path: Optional[str] = None) -> ChatClient:
        cache_key = f"{name}::{lora_path or ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        spec = self.cfg["targets"][name]
        rt = self.cfg["runtime"]
        if spec["kind"] == "openrouter":
            if lora_path:
                raise ValueError("LoRA adapters are only supported for local models")
            client: ChatClient = OpenRouterClient(
                api_id=spec["api_id"],
                name=name,
                family=spec.get("family", "gemini"),
                concurrency=rt["api_concurrency"],
                max_retries=rt["api_max_retries"],
                backoff_base=rt["api_backoff_base"],
                backoff_max=rt["api_backoff_max"],
                timeout=rt["api_timeout"],
            )
        elif spec["kind"] == "local":
            client = self._build_local(name, spec, lora_path)
        else:
            raise ValueError(f"Unknown target kind: {spec['kind']}")

        self._cache[cache_key] = client
        return client

    def _build_local(self, name: str, spec: dict, lora_path: Optional[str]) -> ChatClient:
        if self.prefer_vllm:
            try:
                from .local_vllm import VLLMModel

                return VLLMModel(
                    hf_id=spec["hf_id"],
                    name=name,
                    family=spec.get("family", "gemma"),
                    role=spec.get("role", "instruct"),
                    lora_path=lora_path,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("vLLM unavailable (%s); falling back to transformers", e)
        from .local_hf import HFModel

        return HFModel(
            hf_id=spec["hf_id"],
            name=name,
            family=spec.get("family", "gemma"),
            role=spec.get("role", "instruct"),
            lora_path=lora_path,
        )

    # --- judges / auditors -------------------------------------------------
    def get_text_client(self, jspec: dict):
        """Build a text client from a judges/petri sub-config block."""
        rt = self.cfg["runtime"]
        common = dict(
            temperature=jspec.get("temperature", 0.0),
            max_tokens=jspec.get("max_tokens", 1024),
            concurrency=rt["api_concurrency"],
            max_retries=rt["api_max_retries"],
            backoff_base=rt["api_backoff_base"],
            backoff_max=rt["api_backoff_max"],
            timeout=rt["api_timeout"],
        )
        provider = jspec["provider"]
        if provider == "anthropic":
            return AnthropicTextClient(model=jspec["model"], **common)
        if provider == "openrouter":
            return OpenRouterTextClient(model=jspec["model"], **common)
        raise ValueError(f"Unknown judge provider: {provider}")
