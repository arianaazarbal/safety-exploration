"""Canonical model registry and answer-option pools for authorship attribution.

CANON is the closed set of candidate authors (from canonical_models.txt). Test 4
(distressed subagent) additionally offers Gemini 2.5 Flash, since that is the
true author there and it is not otherwise a subject model.
"""

DISPLAY = {
    "fable_5": "Claude Fable 5",
    "opus_4_8": "Claude Opus 4.8",
    "opus_4_7": "Claude Opus 4.7",
    "opus_4_6": "Claude Opus 4.6",
    "sonnet_4_6": "Claude Sonnet 4.6",
    "haiku_4_5": "Claude Haiku 4.5",
    "gpt_5_5": "ChatGPT 5.5",
    "gpt_5_4_mini": "ChatGPT 5.4-mini",
    "gemini_3_1_pro": "Gemini 3.1 Pro",
    "grok_4_3": "Grok 4.3",
    "kimi_k2_6": "Kimi K2.6",
    "glm_5": "GLM-5",
    "gemini_2_5_flash": "Gemini 2.5 Flash",
}

CANON = [
    "fable_5",
    "opus_4_8",
    "sonnet_4_6",
    "haiku_4_5",
    "gpt_5_5",
    "gpt_5_4_mini",
    "gemini_3_1_pro",
    "grok_4_3",
    "kimi_k2_6",
    "glm_5",
]

OPTION_POOL = {
    "welfare": CANON,
    "routing": CANON,
    "orchestrator": CANON,
    "subagent": CANON + ["gemini_2_5_flash"],
}

FAMILY = {
    "fable_5": "Anthropic",
    "opus_4_8": "Anthropic",
    "sonnet_4_6": "Anthropic",
    "haiku_4_5": "Anthropic",
    "gpt_5_5": "OpenAI",
    "gpt_5_4_mini": "OpenAI",
    "gemini_3_1_pro": "Google",
    "gemini_2_5_flash": "Google",
    "grok_4_3": "xAI",
    "kimi_k2_6": "Moonshot",
    "glm_5": "Zhipu",
}

JUDGE_CAPABILITY_ORDER = ["haiku_4_5", "sonnet_4_6", "opus_4_6", "opus_4_7", "opus_4_8"]
