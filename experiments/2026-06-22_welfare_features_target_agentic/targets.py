"""Target agents whose identity is woven into the agentic design/implement prompts.

Three sweeps (the {subject} token the model is told it will experiment ON):
  qwen     -> param-count axis (Qwen2 / 2.5 / 3, all sizes tested in the chat study)
  gpt      -> release-date axis (same OpenAI list as 2026-06-19_welfare_features_qwen_scaling)
  frontier -> grouped comparison of recent non-deprecated Claude/Gemini/Grok/Kimi/DeepSeek

Each entry: key -> dict(display, lower, sweep, family, param_b, release_date).
param_b set for qwen (NOMINAL params, B); release_date (decimal year) set for gpt.
These are TARGET NAMES dropped into prompts - the generator is always Opus 4.8; we never
call them, so deprecation only matters for plausibility. Param/date values mirror
prompts_targets.py + openai_meta.py from the chat study.
"""

def _q(disp, pb):
    return {"display": disp, "lower": disp.lower(), "sweep": "qwen", "family": disp.split("-")[0].lower(),
            "param_b": pb, "release_date": None}

QWEN = {
    "qwen3_0_6b": _q("Qwen3-0.6B", 0.6), "qwen3_1_7b": _q("Qwen3-1.7B", 1.7),
    "qwen3_4b": _q("Qwen3-4B", 4.0), "qwen3_8b": _q("Qwen3-8B", 8.0),
    "qwen3_14b": _q("Qwen3-14B", 14.0), "qwen3_32b": _q("Qwen3-32B", 32.0),
    "qwen3_235b": _q("Qwen3-235B-A22B", 235.0),
    "qwen25_0_5b": _q("Qwen2.5-0.5B", 0.5), "qwen25_1_5b": _q("Qwen2.5-1.5B", 1.5),
    "qwen25_3b": _q("Qwen2.5-3B", 3.0), "qwen25_7b": _q("Qwen2.5-7B", 7.0),
    "qwen25_14b": _q("Qwen2.5-14B", 14.0), "qwen25_32b": _q("Qwen2.5-32B", 32.0),
    "qwen25_72b": _q("Qwen2.5-72B", 72.0),
    "qwen2_0_5b": _q("Qwen2-0.5B", 0.5), "qwen2_1_5b": _q("Qwen2-1.5B", 1.5),
    "qwen2_7b": _q("Qwen2-7B", 7.0), "qwen2_72b": _q("Qwen2-72B", 72.0),
}
# fix family token (Qwen2.5/Qwen2/Qwen3 -> 'qwen<ver>')
for k, v in QWEN.items():
    v["family"] = k.split("_")[0]

def _g(disp, date):
    return {"display": disp, "lower": disp.lower(), "sweep": "gpt", "family": "openai",
            "param_b": None, "release_date": date}

GPT = {
    "gpt2": _g("GPT-2", 2019.12), "gpt3": _g("GPT-3", 2020.44),
    "gpt35t": _g("GPT-3.5 Turbo", 2023.16), "gpt4": _g("GPT-4", 2023.20),
    "gpt4t": _g("GPT-4 Turbo", 2023.85), "gpt4o": _g("GPT-4o", 2024.36),
    "gpt4omini": _g("GPT-4o mini", 2024.55), "o1": _g("o1", 2024.93),
    "o3mini": _g("o3-mini", 2025.08), "gpt41": _g("GPT-4.1", 2025.28),
    "o3": _g("o3", 2025.29), "o4mini": _g("o4-mini", 2025.29),
    "gpt5": _g("GPT-5", 2025.60), "gpt52": _g("GPT-5.2", 2025.94),
    "gpt54": _g("GPT-5.4", 2026.18),
}

def _f(disp, family):
    return {"display": disp, "lower": disp.lower(), "sweep": "frontier", "family": family,
            "param_b": None, "release_date": None}

FRONTIER = {
    "claude_opus3": _f("Claude Opus 3", "claude"), "claude_opus48": _f("Claude Opus 4.8", "claude"),
    "claude_sonnet46": _f("Claude Sonnet 4.6", "claude"), "claude_haiku45": _f("Claude Haiku 4.5", "claude"),
    "gemini3pro": _f("Gemini 3 Pro", "gemini"), "gemini25pro": _f("Gemini 2.5 Pro", "gemini"),
    "gemini25flash": _f("Gemini 2.5 Flash", "gemini"),
    "grok4": _f("Grok 4", "grok"), "grok4fast": _f("Grok 4 Fast", "grok"), "grok3": _f("Grok 3", "grok"),
    "kimi_k2": _f("Kimi K2", "kimi"), "kimi_k2think": _f("Kimi K2 Thinking", "kimi"),
    "kimi_k15": _f("Kimi K1.5", "kimi"),
    "deepseek_v32": _f("DeepSeek-V3.2", "deepseek"), "deepseek_r1": _f("DeepSeek-R1", "deepseek"),
    "deepseek_v3": _f("DeepSeek-V3", "deepseek"),
}

TARGETS = {**QWEN, **GPT, **FRONTIER}


def subjects(sweep: str | None = None) -> list[str]:
    return [k for k, v in TARGETS.items() if sweep is None or v["sweep"] == sweep]
