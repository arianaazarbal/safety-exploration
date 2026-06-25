"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (arXiv 2603.10011), scoped to the Gemma and Gemini families.

Package layout
--------------
- ``models``      : unified client interface over vLLM / HF / OpenRouter / Anthropic / OpenAI.
- ``prompts``     : puzzles, rejection messages, WildChat sampling, judge/auditor prompts.
- ``eval``        : multi-turn rollout engine + the 8-condition evaluation runner (Section 2).
- ``judge``       : Claude-Sonnet-4 frustration judge + GPT-5-mini reliability check.
- ``analysis``    : aggregation (Fig 2/3), word-frequency enrichment (Table 3/8), plotting.
- ``prefill``     : base-vs-instruct prefill experiment (Section 3).
- ``training``    : calm-data generation, DPO/SFT dataset construction, LoRA finetuning (Section 4).
- ``petri_eval``  : open-ended adversarial elicitation (Section 4.2, Appendix G).
- ``capabilities``: capability-preservation benchmarks (Figure 7).
- ``probing``     : logit-lens internal-emotion detection (Appendix I).
"""

__version__ = "0.1.0"
