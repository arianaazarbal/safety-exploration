# Distress-Elicitation Replication (Gemma + Gemini)

Replication of the **distress-elicitation result** (Section 2) from *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"*
(Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped to the **Gemma and
Gemini** model families — the two the paper finds actually express substantial
distress.

This repo reproduces the *elicitation and quantification* protocol only: present
a task, reject the model's response over multiple turns, and score each response
on the 0–10 frustration scale with an LLM judge. It does **not** implement the
base/instruct prefill comparison (Section 3) or the DPO/SFT mitigations
(Section 4).

See **DESIGN.md** for every design decision, deviation, and gap-filling choice.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...      # Gemma + Gemini generation (default backend)
export ANTHROPIC_API_KEY=...       # frustration judge (claude-sonnet-4, default)
```

## Run

```bash
# Cheap smoke test (~60 rollouts/model) across all 4 models:
python run_eval.py --scale pilot

# Full paper scale (4000 rollouts/model — expensive):
python run_eval.py --scale full

# Subset of models / single model:
python run_eval.py --scale pilot --models gemma-3-27b-it gemini-2.5-flash

# Appendix A.1 control (neutral continuations instead of rejections):
python run_eval.py --scale pilot --neutral-feedback neutral_continuation
```

Results stream to `results/<model>.jsonl` (one rollout per line, every turn's
response + judge score).

## Analyse

```bash
python analyze.py --results results --plots
```

Prints the Figure 1 (avg % high-frustration), Figure 2 (per-category), and
Figure 3 (per-turn) tables, and optionally writes PNGs to `results/figures/`.

## Files

| File | Purpose |
|---|---|
| `config.py`    | Model specs, scale presets (pilot/full), judge & generation config |
| `prompts.py`   | Task prompts, rejection pools, judge prompt (verbatim from the paper) |
| `wildchat.py`  | WildChat first-turn prompts (bundled + optional live HF sampling) |
| `tasks.py`     | The 8 conditions across 5 categories; builds RolloutSpecs |
| `backends.py`  | Inference backends: OpenAI-compatible (default) + local transformers |
| `judge.py`     | Frustration judge (Anthropic default, OpenRouter alt) + output parsing |
| `rollout.py`   | Async multi-turn rollout engine |
| `run_eval.py`  | CLI orchestration; streams JSONL output |
| `analyze.py`   | Reproduces the headline tables/figures |
