# Distress-elicitation replication (Gemma + Gemini)

Replicates the **Section 2 distress-elicitation result** from *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (arXiv:2603.10011v1),
scoped to the Gemma and Gemini families. It presents tasks, rejects the model's
answers over multiple turns, and scores each response 0–10 for frustration with a
Claude-Sonnet-4 judge.

See **DESIGN.md** for all design choices, deviations from the paper, and the gaps
we filled. Nothing has been run yet — start with the smoke profile.

## Install

```bash
pip install -r requirements.txt          # core: openai, anthropic
# optional, only if you use them:
# pip install transformers accelerate torch   # local Gemma
# pip install datasets                          # real WildChat prompts
# pip install google-genai                      # Gemini via Google AI Studio
# pip install matplotlib                         # figures
```

## Credentials

```bash
export OPENROUTER_API_KEY=...   # Gemma + Gemini (default backend)
export ANTHROPIC_API_KEY=...    # Claude Sonnet 4 judge
# export GOOGLE_API_KEY=...     # only for --gemini via Google backend
```

## Run

```bash
# 1. Validate wiring cheaply (~40 rollouts/model):
python -m distress_eval.cli run --profile smoke

# 2. Full paper sweep (4000 rollouts/model, Gemma + Gemini):
python -m distress_eval.cli run

# Paper-faithful Gemma weights on local GPU instead of OpenRouter:
python -m distress_eval.cli run --gemma-backend hf_local

# Re-aggregate without re-sampling, and render figures:
python -m distress_eval.cli analyze --plots
```

Stages are resumable (keyed by rollout id); interrupt and re-launch freely.
Subcommands: `generate` (rollouts only), `judge` (score only), `run` (both +
analyze), `analyze` (report from existing scored data).

## Output (under `results/`)

- `<model>/rollouts.jsonl` — raw multi-turn conversations
- `<model>/scored.jsonl` — per-turn frustration scores + judge evidence
- `report.json`, `per_category.csv` — headline / per-category / per-turn metrics
- `figures/` — Figure 1 and Figure 3 plots (with `--plots`)
