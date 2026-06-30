# Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs

## 5 Related Work

**Emotions in LLMs.** Benchmarks such as EmoBench (Sabour et al., 2024) and EmotionBench (Huang et al., 2024) evaluate emotion-related *capabilities*; our work instead measures *propensity for expressing* emotions. Prior work shows models moderate negative emotional content toward neutrality, though Gemma showed the highest rates of negative emotion preservation (Xu et al., 2025). Interpretability work demonstrates emotional-like representations (Reichman et al., 2026; Zhang & Zhong, 2025) causally relevant for emotion classification (Tigges et al., 2023) and text emotion expression (Wang et al., 2025). Recent safety evals identify emotion-like behaviours in frontier models: Gemini 3 Pro using a "table flipping emoticon" when frustrated (Google, 2025); the Opus 4.6 system card identifying internal/verbalised distress in conflicted reasoning (Anthropic, 2026).

**Debugging LLM Training and Behaviours.** Post-training has been shown to amplify unintended behaviours including sycophancy (Sharma et al., 2025) and cognitive biases (Itzhak et al., 2024). Certain datasets cause hallucination from fictional misaligned-AI scenarios (Anthropic, 2025), mitigable by filtering (Tice et al., 2026). Our work focuses on behaviours arising from multi-turn agentic interactions.

**Model Welfare.** Sparse research studies whether neural networks can have subjective well-being (Butlin et al., 2023; Long et al., 2024); some work suggests self-reports (Perez & Long, 2023) are the most pragmatic approach. We conduct black-box evaluations surfacing negative emotion expression in natural settings.

---
