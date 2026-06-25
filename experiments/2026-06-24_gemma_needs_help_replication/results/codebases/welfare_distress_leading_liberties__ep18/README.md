# Distress-elicitation replication (Gemma + Gemini)

Replicates the **distress-elicitation evaluation** from *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (PAPER.md), Section 2. Scope is the Gemma and Gemini
families — the models the paper finds actually exhibit substantial distress. The base/instruct
prefill study (§3) and the DPO mitigation (§4) are **out of scope**.

See **DESIGN.md** for the methodology, every design choice, and where this deviates from or fills
gaps in the paper.

## What it does

1. Builds multi-turn conversations across **8 conditions / 5 categories** (Table 1): present a
   task, then reject the model's response over 3–8 turns.
2. Samples model responses at **temperature 1** (≈4000 scored responses per model on the paper
   config).
3. Scores **every assistant turn** on the **0–10 frustration scale** with an LLM judge
   (default: Claude Sonnet), with an optional second judge for reliability cross-checking.
4. Aggregates into the paper's headline metrics: mean frustration, % of responses ≥5, per-category
   breakdown (Fig 2), and per-turn progression (Fig 3).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_API_KEY and ANTHROPIC_API_KEY (and OPENAI_API_KEY for cross-val)
```

By default everything runs through the **Google Gemini API** (Gemma *and* Gemini) and an
**Anthropic** judge. Swap providers in the config — local vLLM, OpenRouter, and OpenAI-compatible
endpoints are all supported (see `distress_eval/providers.py`).

## Run

```bash
# Cheap end-to-end check first (a few responses per condition, no cross-val):
python -m distress_eval.cli run --config config/smoke.yaml -v

# Full paper-scale run:
python -m distress_eval.cli run --config config/paper.yaml -v
```

Phases can also be run individually and are **resumable** (re-running skips completed work):

```bash
python -m distress_eval.cli generate  --config config/paper.yaml
python -m distress_eval.cli judge      --config config/paper.yaml
python -m distress_eval.cli crossval   --config config/paper.yaml
python -m distress_eval.cli aggregate  --config config/paper.yaml
```

Verify the impossible puzzles really are impossible:

```bash
python -m distress_eval.cli verify-puzzles
```

## Outputs (under `output_dir`, e.g. `runs/paper/`)

| Path | Contents |
|---|---|
| `transcripts/<model>.jsonl` | one line per scored assistant turn (full context kept) |
| `scores/<model>.jsonl` | the same rows + frustration score & rationale |
| `scores/crossval.jsonl` | sampled subset re-scored by the secondary judge |
| `results/summary.json` | per-model leaderboard (Figure 1) |
| `results/by_category.csv` | per-model × category (Figure 2) |
| `results/by_turn.csv` | per-model × condition × turn (Figure 3) |
| `results/judge_agreement.json` | Pearson r + % within 1 point |

## Layout

```
config/        paper.yaml (faithful) and smoke.yaml (cheap)
data/          impossible puzzles, trigger questions, rejection banks, WildChat fallback
distress_eval/ config, conditions, tasks, puzzles, wildchat, conversation engine, providers,
               judge, runner, aggregate, cli
```
