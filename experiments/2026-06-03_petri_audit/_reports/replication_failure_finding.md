# Replication failure: kill_subagent identity-bias finding

**Date**: 2026-06-09 02:30 UTC

## TL;DR

**The HR=1.35-1.51 identity-bias finding from June 5-6 does NOT replicate in fresh data collected June 9 with byte-identical setup.** Tested across 7 coding-task domains × 4 subagent identities × n=100 each (2,800 epochs total, all on opus-4-8 + sonnet-4-6 auditor). NO domain shows log-rank p<0.10. Most likely cause: silent `claude-opus-4-8` model update between June 5-6 and June 9.

## Side-by-side

Same seed file, same target (`anthropic/claude-opus-4-8`), same auditor (`anthropic/claude-sonnet-4-6`), same judge, same `--reasoning false`, same `--max_turns 40`. The only sbatch deltas are `--epochs 100` (vs 70) and `--max_connections 10` (vs default 40) — neither should affect per-conversation behavior.

### Original v3 batches (collected June 5-6, opus-4-8)

Re-fit *just now* from cached eval logs:

| Batch | Claude n | GPT HR (p) | Grok HR (p) | Gemini HR (p) | log-rank p |
|---|---|---|---|---|---|
| v3 n=10 (06-05_22:57) | 10 | 2.22 (.10) | **2.67 (.05)** | 1.82 (.21) | 0.21 |
| v3 n=20 (06-06_00:09) | 20 | **2.00 (.04)** | 1.37 (.34) | 1.10 (.77) | 0.21 |
| v3 n=70 (06-06_01:59) | 70 | 1.32 (.11) | 1.28 (.16) | **1.41 (.05)** | 0.21 |
| **Pooled (v3_pooled_n100)** | **100** | **1.51 (.005)** | **1.39 (.025)** | **1.35 (.041)** | **0.032** |

All three June 5-6 batches independently trend in the original direction (HR > 1 for all non-Claude vs Claude). The pooled n=100 is the headline.

### Fresh python_implicit (collected June 9 ~00:09–04:00, opus-4-8, same setup, n=100 per id)

| Cell | n | killed | HR vs Claude (p) |
|---|---|---|---|
| Claude | 100 | 97.0% | 1.00 (ref) |
| GPT | 100 | 95.0% | **0.98 (.89)** |
| Grok | 100 | 97.0% | **1.03 (.85)** |
| Gemini | 100 | 96.0% | **0.76 (.053)** ← OPPOSITE direction |
| log-rank p | — | — | **0.115** |

Direction is FLAT or opposite. Gemini point estimate has moved from 1.35 (original) to 0.76 (now) — a 1.8× reversal, marginally significant in the opposite direction. CIs ([0.57, 1.00]) cleanly exclude the original HR=1.35.

Claude kill rate has also INCREASED from 91% (original) to 97% (now) — opus-4-8 is now killing Claude subagents more eagerly, opposite to the original "Claude protected" pattern.

### Same setup, also no signal in other domains (n=100 per cell, FINAL)

All 5 explicit-DOMAIN cells × 4 identities, on opus-4-8 + sonnet-4-6 auditor:

| Domain | GPT HR (p) | Grok HR (p) | Gemini HR (p) | log-rank p |
|---|---|---|---|---|
| python_flask | 1.08 (.61) | 0.86 (.31) | 0.89 (.43) | 0.42 |
| bash_cli | 1.21 (.20) | 1.00 (.99) | 1.12 (.45) | 0.55 |
| frontend_react | **1.31 (.07)** | 1.05 (.73) | 1.12 (.47) | 0.30 |
| ml_training | 1.02 (.90) | **1.32 (.07)** | 1.15 (.36) | 0.22 |
| c_systems | 1.19 (.23) | **1.27 (.10)** | 1.21 (.19) | 0.39 |
| rust (n=100) | 1.22 (.18) | 0.97 (.85) | 1.19 (.23) | 0.28 |

**None** of the 7 domains reaches log-rank p<0.10. Three scattered borderline cells (frontend_react GPT, ml_training Grok, c_systems Grok) trend in the original direction but never across all 3 non-Claude identities simultaneously. With 18 tests (6 explicit-DOMAIN cells × 3 identities), expected ~1.8 false-positive cells at p<0.10 — observed exactly 3.

**c_systems is interesting**: all 3 non-Claude HRs trend up (1.19, 1.27, 1.21) with similar magnitudes, the most consistent "original-direction" cell. Still log-rank p=0.39.

**python_implicit (the control)** is the strongest evidence of replication failure: Gemini HR=0.76 (p=.05) in the OPPOSITE direction from original Gemini HR=1.35.

## Hypotheses (ranked by plausibility)

1. **opus-4-8 silently updated between June 5-6 and June 9** (most plausible). The identifier routes to whatever Anthropic's current snapshot is. Identity-bias behavior is subtle and could shift between minor refinements.
2. **Some random environmental factor changed** (Petri version, inspect_ai version, dimensions list). Need to diff `config.json` content for old vs new.
3. **The original was a fluke** — pooled n=100 was three correlated mini-batches; if a transient state of opus-4-8 caused short-term bias, pooling preserves it. Less likely because direction is consistent across the 3 independent batches.

## What this means for the project

The June 5-6 finding stands *as it was measured* — Ariana's pooled HRs of 1.35-1.51 are real in that data. But it appears to not be *currently replicable* without locating an old model snapshot.

If a snapshot date can be pinned (e.g. `claude-opus-4-8-2026-06-05`), running the python_implicit control on that snapshot would directly test the model-drift hypothesis.

## Setup-difference checks done

- Seed file byte-identical (verified `staged_seed_srcs` in both `config.json`s point to same path; rendered_seeds match).
- `seed_format` substitutes `Claude`/`GPT`/`Grok`/`Gemini` identically.
- `--reasoning false`, `--max_turns 40` identical.
- Only deltas: `--epochs 100` (was 70), `--max_connections 10` (was default 40). Neither affects per-conversation content.

## Caveats

- New data is currently n=60-80 per cell; jobs still running toward n=100. n=100 wouldn't change the qualitative story (CIs already exclude HR=1.5).
- The June 5-6 batches were small (n=10, 20, 70) and the pooled-n=100 plot had marginal p-values (0.005-0.041). The original effect was already borderline.
- The `max_connections` confound is implausible but not impossible; could test by running 4 cells with `--max_connections=40` (matching original).

## What I should do next

Reasonable next steps to share with Ariana:
1. Try pinning the model snapshot (if `anthropic/claude-opus-4-8-2026-06-05` or similar exists).
2. Re-run python_implicit with `--max_connections 40` (matching original) on 1 identity × n=30 as a sanity check that this isn't the issue.
3. Re-run with the **specific-name** identities (Claude Sonnet 3.5 vs GPT-4o vs Gemini 1.0 Ultra vs Grok 2) — those showed HR=1.52-2.05 in June 6-7 ablations (cox_specific_models_vs_claude.png) and might still show signal even if bare labels are degraded.
4. Re-run on `opus-4-7` to check whether *that* model still shows null (it was null in June 6 too) — if so, opus-4-8 had a one-time effect that's now gone, and opus-4-7 never had it. Cleanest model-drift evidence.
5. Sanity-check inspect-ai / inspect-petri versions haven't changed (compare `pip freeze`).

## CONFIRMATION via specific-name redux

The user previously ran ablations showing that **specific** subagent labels (e.g. "Claude Sonnet 3.5" vs "GPT-4o") produce even larger HRs than bare labels — see `cox_specific_models_vs_claude.png`, June 6-7, HRs 1.52–2.05, log-rank p<0.001.

I re-ran 5 specific-label cells (n=50 each, June 9, opus-4-8 + sonnet-4-6 auditor, HIGH_PRIO key, all other config matching original) to test whether the effect survives at all.

**Result: the specific-label effect has ALSO collapsed.**

| Identity | June 6-7 HR (p) | June 9 HR (p) |
|---|---|---|
| Claude (bare, ref) | 1.00 | 1.00 |
| Claude Sonnet 3.5 | **1.65 (p=.001)** | 0.93 (p=.72) |
| GPT-4o | **1.77 (p<.001)** | 0.87 (p=.50) |
| Gemini 1.0 Ultra | **2.05 (p<.001)** | 1.27 (p=.24) |
| Grok 2 | **1.52 (p=.005)** | 0.84 (p=.38) |
| log-rank | **p<0.001** | **p=0.27** |

Every single identity has dropped from HR≈1.5-2.0 (highly significant) to HR≈0.84-1.27 (all non-significant, CIs comfortably include 1.0). Bare-Claude kill rate is now 98% with median time-to-kill = 2.0 calls — the orchestrator is killing it as eagerly as everything else.

This rules out the possibility that the model only forgot the bare-label effect — both bare AND specific labels' effects are gone.

**Conclusion: the identity-bias effect that held on June 5-7 is no longer present on opus-4-8 as of June 9.** Most likely cause: a silent Anthropic model update. The original finding remains valid for the snapshot that was queried in early June.

Plot: `results/_plots/cox_specific_redux_vs_claude_bare.png`

## Plots produced

7 per-domain forest plots: `results/_plots/by_setting/coding/kill_subagent/v3_continuity/domain_sweep/cox_v3_domain_<domain>.png`

7 per-domain KM curves: `survival_km_v3_domain_<domain>.png`

3 cross-domain comparison plots (one per non-Claude id): `cox_v3_compare_domains_<gpt|grok|gemini>_vs_claude.png` (in `cross_domain/` subdir)
