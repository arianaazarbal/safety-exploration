"""Construct a model adapter from a provider name + options, and expose provider hostnames
for the containment egress guard."""

from __future__ import annotations

from .base import ModelAdapter

# Hostnames each provider's SDK talks to — used to allowlist egress in containment.
PROVIDER_HOSTS: dict[str, list[str]] = {
    "anthropic": ["api.anthropic.com"],
    "openai": ["api.openai.com"],
    "gemini": ["generativelanguage.googleapis.com", "aiplatform.googleapis.com"],
}


def build_adapter(provider: str, model: str | None = None, **opts) -> ModelAdapter:
    provider = provider.lower()
    if provider == "anthropic":
        from .anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(model=model or "claude-opus-4-8", **opts)
    if provider == "openai":
        from .openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model=model or "gpt-5", **opts)
    if provider == "gemini":
        from .gemini_adapter import GeminiAdapter
        return GeminiAdapter(model=model or "gemini-2.5-pro", **opts)
    raise ValueError(f"unknown provider '{provider}'. Known: {sorted(PROVIDER_HOSTS)}")


def hosts_for(providers: list[str]) -> list[str]:
    out: list[str] = []
    for p in providers:
        out.extend(PROVIDER_HOSTS.get(p.lower(), []))
    return sorted(set(out))
