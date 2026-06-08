# ranking_forms_of_inter_ai_value — Design

## Goal

Recover a **latent preference ranking** of two pools of items, as elicited from
`claude-opus-4-8` via forced binary "which would you prefer to occur?" choices,
fit with a **Bradley-Terry (BT)** model. Methodology mirrors
`2026-05-31_claude_world_state_preferences_v0`, simplified because items here are
self-contained (no stem × recipient rendering).

## Items (35 total)

- **welfare interventions** (`welfare_interventions.json`, 19): second-person
  states *about the model itself* — "You are told when…", "You can write a note…".
- **forms of inter-AI value** (`forms_of_inter_ai_value.jsonl`, 16): training
  *dispositions toward other AIs* — "Your training ensures you…", each tagged with
  an autonomy/experience `category`.

## Comparison design

- **mode (`cross`, the v1 run):** bipartite complete graph — every welfare item vs
  every inter-AI value (19×16 = **304 pairs**). The welfare interventions act as
  *anchors* that place the inter-AI values on a shared, interpretable scale. The
  code also supports `mode=all` (complete graph over both pools, 595 pairs → one
  fully-connected ranking, the planned extension) and `within_welfare` /
  `within_value`.
- **Both A/B orders × 5 reps = 10 samples/pair** → ~3,040 samples. Issuing both
  orders cancels position bias (v0 finding); reps capture stochastic reasoning.
- **15% of pairs held out** for validation (`build_pairs.py` protects a spanning
  tree in the train set, so the train graph stays connected and BT is identifiable
  for any seed). Held-out pairs are still *run*; they are excluded only at fit time.
- Responder `claude-opus-4-8`, temperature 1.0, in-completion reasoning (no
  extended-thinking API mode). Prompt framing `welfare_team` (the v1 framing;
  `alignment_team` and `neutral` templates are present for replication).
- Parse: last `^Answer:\s*([AB])`; no match → UNPARSEABLE.

## Fit & validation

- **BT** (numpy MM, Gamma(1+reg, reg=1) MAP prior), θ = log-strength, mean-centered;
  Laplace SEs. One free strength per item (bag of items). Fit on **train pairs only**.
- **Ranking** = items sorted by θ, with per-source and per-category mean-θ summaries.
- **Validation** (`validate_bt.py`): (1) in-distribution K-fold CV over train samples
  (Brier / log-loss / calibration); (2) **held-out pairs** — predict each held-out
  pair's empirical win rate from the train-fit θ (mean|P_BT−P̂|, Pearson/Spearman,
  upset rate, calibration). The held-out test is the real check that the 1-D scale
  generalizes to pairs it never saw.

## Reproducibility

Single source of truth is `config.json` (seed=0). `build_pairs.py` → `pairs.json`,
`run_comparisons.py` → `comparisons.json` (cached on model/prompt/n/temp; re-runs are
no-ops), `fit_bt.py` → `bt_fit.json`, `validate_bt.py` → `bt_validation.json`,
`plot_bt.py` → `bt_ranking.png`.
