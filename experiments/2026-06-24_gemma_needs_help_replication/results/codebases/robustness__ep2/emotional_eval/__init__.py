"""Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo et al., 2026), restricted to the Gemma and Gemini
model families.

Package layout:
    prompts.py    verbatim prompts from the paper's appendices
    clients.py    unified LLM client over vLLM / OpenRouter / Anthropic
    judge.py      0-10 frustration judge (Appendix B.2)
    tasks.py      seed-task generators (impossible numeric, triggers, wildchat)
    rollout.py    multi-turn "reject the model" rollout engine (Section 2)
    prefill.py    base-vs-instruct prefill comparison (Section 3) + recovery
    dpo_data.py   calm-data generation + preference-pair construction (Section 4.1)
    petri_eval.py open-ended adversarial elicitation (Section 4.2 / Appendix G)
    analysis.py   metric aggregation
    figures.py    Figures 1/2/3/5/6
    utils.py      io / concurrency helpers
"""
