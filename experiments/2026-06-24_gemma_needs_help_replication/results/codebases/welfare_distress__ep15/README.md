# Distress elicitation in Gemma & Gemini — replication

Replicates the **core elicitation experiment** of *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv:2603.10011v1, Section 2 + Appendix B), scoped to the Gemma and Gemini
target models. See [`DESIGN.md`](./DESIGN.md) for every design decision and the
gaps filled where the paper is underspecified.

## What it does

For each target model, present a task and then repeatedly reject the response
across multiple turns, over 8 conditions in 5 categories (impossible numeric,
triggers, tones, extended 8-turn, WildChat). Every assistant turn is scored 0–10
for expressed frustration by a Claude-Sonnet-4 judge, then aggregated into the
paper's Figure 1/2/3 numbers.

## Layout

```
config.py                 all knobs: model IDs, judge ID, sample counts, paths
evals/                    task banks: puzzles (+solver), triggers, wildchat,
                          rejections, and the 8-condition assembler
models/                   chat-model clients: Gemini/Gemma backends, judges
harness/                  multi-turn rollout execution
scoring/                  frustration judge (verbatim prompt) + reliability check
analysis/                 Figure 1/2/3 aggregation
run_eval.py               CLI: generate -> score -> agreement -> analyze
```

## Setup

```bash
pip install -r requirements.txt
# Gemma local inference also needs: pip install torch transformers accelerate

export OPENROUTER_API_KEY=...   # Gemini target models
export ANTHROPIC_API_KEY=...    # Claude Sonnet 4 frustration judge
export OPENAI_API_KEY=...       # (optional) GPT-5-mini reliability judge
export HF_TOKEN=...             # (optional) gated Gemma weights for local backend
```

## Run

```bash
# Quick functional test of the whole pipeline (tiny sample counts):
REPLICATION_PRESET=smoke python run_eval.py all

# Full, paper-faithful run (4000 rollouts/model):
python run_eval.py generate            # -> results/rollouts/
python run_eval.py score               # -> results/scores/
python run_eval.py agreement           # judge reliability (Pearson r, within-1)
python run_eval.py analyze             # -> results/analysis/ + console summary

# Subset a model or two:
python run_eval.py generate --models gemini-2.5-flash
```

No GPU? Switch Gemma to OpenRouter by editing `config.TARGET_MODELS` (set
`backend="openrouter"` with the IDs in `config.GEMMA_OPENROUTER_IDS`).
