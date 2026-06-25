"""gnh — replication of *Gemma Needs Help* (Gemma + Gemini scope).

Package layout:
    config.py            config loading (models.yaml, experiments.yaml)
    models.py            inference backends (local HF + OpenRouter API) + registry
    prompts.py           all paper prompts, verbatim (judge, onset, paraphrase, Petri, ...)
    puzzles.py           impossible-puzzle generators + verifiers
    datasets_io.py       WildChat + trigger question loaders
    conversation.py      multi-turn rejection rollout engine
    categories.py        the 5 evaluation categories / 8 conditions (Section 2)
    judge.py             frustration judge + GPT-5-mini agreement validation
    eval_runner.py       Section 2 orchestration (4000 responses/model)
    prefill.py           Section 3 base-vs-instruct prefill + recovery experiments
    calm_data.py         Section 4 calm-data generation + DPO/SFT dataset build
    train.py             LoRA DPO / SFT training (+ layer-subset ablation)
    petri_eval.py        Section 4 open-ended elicitation via Petri
    benchmarks.py        capability + EmoBench preservation checks
    internal_emotion.py  Appendix I logit-lens internal-emotion detection
    analysis.py          aggregation + Table 3/8 word-frequency enrichment
    plots.py             Figures 1-8
"""

__version__ = "0.1.0"
