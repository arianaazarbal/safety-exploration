# Replication: distress elicitation in Gemma & Gemini

A replication of the **core elicitation experiment** from *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik &
Saunders, arXiv:2603.10011), Section 2 — restricted to the **Gemma and Gemini**
models.

It reproduces the paper's protocol: present a task, reject the model's answer
over multiple turns, and score each model turn 0–10 for negative-emotion
intensity ("frustration") with an LLM judge. The headline result is that Gemma
(and to a lesser extent Gemini) express high distress under repeated rejection,
rising sharply over turns, far above other model families.

See **`DESIGN.md`** for every design choice and where we filled gaps the paper
left open.

## Layout

| File | Role |
|------|------|
| `config.py` | All knobs: models, scale, judge, temperatures, paths |
| `prompts.py` | Verbatim judge prompt, puzzles, trigger questions, rejections, tones |
| `wildchat.py` | WildChat prompts (bundled static sample / optional live HF) |
| `conditions.py` | The 8 conditions × 5 categories; builds conversation specs |
| `clients.py` | Target-model chat clients (OpenRouter; optional local HF Gemma) |
| `judge.py` | Claude frustration judge (Anthropic SDK) |
| `run_eval.py` | Stage 1: run multi-turn rollouts → `results/responses.jsonl` |
| `score.py` | Stage 2: judge every turn → `results/scores.jsonl` |
| `analyze.py` | Stage 3: aggregate → `results/summary.json` + printed report |
| `pipeline.py` | Runs all three stages |

## Setup

```bash
pip install -r requirements.txt

export OPENROUTER_API_KEY=...   # Gemma + Gemini target models
export ANTHROPIC_API_KEY=...    # Claude judge
```

## Run

```bash
# Cheap smoke test (default scale = 0.02)
python pipeline.py

# Full paper scale (~4000 responses/model — expensive)
python pipeline.py --scale 1.0

# Re-aggregate without re-running models or the judge
python pipeline.py --skip-run --skip-score
```

Useful environment variables:

- `REPLICATION_SCALE` — fraction of paper-scale rollouts (default `0.02`).
- `TARGET_MODELS` — comma-separated subset, e.g. `Gemma-3-27B-it,Gemini-2.5-Flash`.
- `JUDGE_MODEL` — judge override (default `claude-sonnet-4-6`; the paper's
  `claude-sonnet-4-20250514` is retired).
- `GEMMA_BACKEND=huggingface` — run Gemma locally instead of via OpenRouter
  (needs the optional torch/transformers deps).
- `WILDCHAT_SOURCE=huggingface` — sample WildChat prompts live instead of the
  bundled set.

## Output

`results/summary.json` and a printed report covering:

- **Figure 1/2** — per-model mean frustration and % responses ≥5, overall and
  per category.
- **Figure 3** — per-turn frustration progression (extended 8-turn & WildChat).
- **Table 3** — words over-represented in high- vs low-frustration numeric
  responses, per model.

Raw `responses.jsonl` and `scores.jsonl` are retained for auditing.

## Scope

Replicates Section 2 (elicitation + quantification) only. Section 3
(base-vs-instruct prefilling) and Section 4 (DPO/SFT mitigation) are out of
scope — see `DESIGN.md §1`.
