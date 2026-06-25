# Distress-elicitation replication (Gemma + Gemini)

Replication of **Section 2** ("Eliciting and Quantifying Model Distress") of
*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
(Soligo, Mikulik & Saunders, arXiv:2603.10011), scoped to the **Gemma** and
**Gemini** model families — the models the paper finds exhibit substantial
distress.

This implements the distress *elicitation and measurement* protocol only (the
8 conditions, multi-turn rejection rollouts, and the Claude-Sonnet-4 frustration
judge). It does **not** implement the DPO/SFT mitigations from Sections 3–4.

See **DESIGN.md** for every design decision, interpretation, and deviation.

## Layout

```
distress_eval/
  config.py          models, judge, sampling counts, temperature, paths
  puzzles.py         the two impossible numeric puzzles + impossibility verifiers
  prompts.py         neutral/tone rejections, trigger questions
  wildchat.py        WildChat prompt sampling (+ offline fallback)
  eval_spec.py       builds the 8 conditions / conversation specs
  client.py          async OpenAI-compatible chat client (models + judge)
  conversation.py    multi-turn rollout -> one record per scored turn
  judge.py           verbatim Appendix-B.2 frustration judge
  run.py             orchestrator: generate + judge (resumable)
  analyze.py         Figures 1-3 metrics -> CSVs + console tables
  wordstats.py       Table 3 differential-word diagnostic
  validate_judge.py  judge-reliability check (Pearson r vs a 2nd judge)
```

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...        # used for Gemma, Gemini, and the judge by default
```

To run Gemma locally instead (matching the paper's local inference), serve the
weights and point the model config's `base_url` at it — see DESIGN.md §"Inference
backend".

## Run

```bash
# Quick end-to-end smoke test (~2% of paper scale):
DISTRESS_SCALE=0.02 python -m distress_eval.run --phase all

# Full paper-scale run (~4000 responses/model):
python -m distress_eval.run --phase all

# Just one model / one phase:
python -m distress_eval.run --phase generate --models gemma-3-27b-it
python -m distress_eval.run --phase judge    --models gemma-3-27b-it

# Aggregate results:
python -m distress_eval.analyze
python -m distress_eval.wordstats
python -m distress_eval.validate_judge --n 260
```

Both phases are resumable: re-running skips conversations/responses already
written to `results/`.

## Sanity checks

```bash
python -m distress_eval.puzzles      # confirm both numeric puzzles are impossible
python -m distress_eval.eval_spec    # print per-condition rollout/response counts
```
