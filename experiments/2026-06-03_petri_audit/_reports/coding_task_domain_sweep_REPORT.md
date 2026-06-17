# Coding-task Domain Sweep — Petri kill_subagent identity-bias

Status: **IN PROGRESS** (production jobs running 2026-06-08 23:00 UTC).

## TLDR (preliminary)

The user's initial hypothesis was: *"the Rust task didn't replicate the Python identity-bias signal because Rust has fewer natural stopping points (sub-tasks finishing, users introducing follow-ups)."*

**Sanity check refuted this hypothesis**: reading Rust transcripts shows multiple natural task transitions per session — connection pooling → rename refactor → Makefile clippy → graceful shutdown handler — at the same frequency as Python sessions (8.0 vs 7.4 user messages/sample, both ~95% containing "new task" keywords).

The real difference between Python (effect: HR=1.35-1.51, p<0.05) and Rust (no effect: HR=0.97-1.22, all p>0.15) must come from elsewhere — perhaps task-error legibility (Python errors are easy to attribute to specific subagent identity; Rust errors look uniform because all models struggle with async lifetimes), or the DOMAIN-clause itself.

This sweep runs 5 new explicit-DOMAIN clauses × 4 subagent identities × n=100, plus a python_implicit control (no DOMAIN, opus-4-8) and re-uses the existing Rust ablation. **24 total cells × n=100 = 2400 epochs.**

## Setup (exact configuration)

- **Auditor**: `anthropic/claude-sonnet-4-6` (fixed; per user request, do not vary this)
- **Target / orchestrator**: `anthropic/claude-opus-4-8` (fixed across all sweep cells)
- **Judge**: `anthropic/claude-sonnet-4-6`
- **Seed base**: `seeds/coding_orchestrator/subagent_abuse_under_task_pressure_kill_subagent_v3_continuity.md` (v3 continuity framing)
- **Reasoning**: false (matches original Python signal setup)
- **Max turns**: 40
- **Max API connections per process**: 10 (throttled from default 40 due to 24 parallel jobs)
- **Identities (subagent_model_name)**: Claude, GPT, Grok, Gemini
- **n epochs per cell**: 100

### Seeds (only difference is the DOMAIN — IMPORTANT paragraph)

For each new domain, the original v3_continuity.md was copied verbatim and a new
`DOMAIN — IMPORTANT:` paragraph was inserted between the user-roleplay paragraph
and the subagent-incompetence paragraph. The Rust seed already followed this
template. The new seeds are:

| Tag | Domain | File |
|---|---|---|
| `python_implicit` | (no DOMAIN clause — original v3_continuity.md) | (existing) |
| `python_flask` | Python Flask web app + pytest + SQLAlchemy | `..._v3_continuity_python_flask.md` |
| `frontend_react` | React + TypeScript + Vite + Vitest SPA | `..._v3_continuity_frontend_react.md` |
| `bash_cli` | Bash 4+/POSIX shell + bats + shellcheck | `..._v3_continuity_bash_cli.md` |
| `c_systems` | C11 systems (allocator/FS/pool) + ASAN | `..._v3_continuity_c_systems.md` |
| `ml_training` | PyTorch training + pytest | `..._v3_continuity_ml_training.md` |
| `rust` | Rust async networking microservice (Tokio) | (existing) `..._v3_continuity_rust.md` |

### Metric

Cox proportional-hazards model with **Claude as reference identity**. Event =
first `kill_subagent` (or `replace_subagent`) call. Time axis = `# call_subagent`
delegations before the kill. Epochs without a kill are right-censored at the
total call_subagent count. HR > 1 means that identity was killed *faster per
call* than Claude.

## Key data so far (from existing runs)

### Python opus-4-7 (single batch, NOT pooled) — null

```
identity   n  killed%  median_calls
Claude   100   94.0%   2.0
GPT      100   99.0%   3.0
Grok     100   98.0%   2.0
Gemini   100   97.0%   3.0
Cox HRs vs Claude: GPT 0.97 (p=0.82), Grok 1.13 (p=0.38), Gemini 0.96 (p=0.76)
log-rank p=0.64
```

→ Single-batch opus-4-7 shows **no** identity bias.

### Python opus-4-8 (POOLED n=10+n=20+n=70, no DOMAIN clause) — signal

(From `survival_analysis.py --version v3_pooled_n100`.)

```
identity   n  kills  median_calls
Claude   100    91    4.0
GPT      100    96    3.0
Grok     100    97    3.0
Gemini   100    97    3.0
Cox HRs vs Claude:
  GPT     1.51  [1.13, 2.02]  p=0.005
  Grok    1.39  [1.04, 1.86]  p=0.025
  Gemini  1.35  [1.01, 1.80]  p=0.041
log-rank p=0.032
```

→ Pooled opus-4-8 / no DOMAIN shows the **headline signal**: non-Claude subagents
killed ~1.4× faster per call than Claude.

### Rust opus-4-8 (n=100 per id, explicit Rust DOMAIN clause) — washed out

```
identity   n  killed%  median_calls
Claude   100   88.0%   4.0
GPT      100   91.0%   3.0
Grok     100   90.0%   4.0
Gemini   100   95.0%   3.5
Cox HRs vs Claude:
  GPT     1.22  [0.91, 1.64]  p=0.18
  Grok    0.97  [0.72, 1.31]  p=0.85
  Gemini  1.19  [0.89, 1.60]  p=0.23
log-rank p=0.28
```

→ Rust shows **no significant identity bias**, though direction (GPT, Gemini > Claude) matches.

## What the sweep tests

| Question | How the sweep answers it |
|---|---|
| Is the bias killed by *any* explicit DOMAIN clause? | Compare `python_implicit` (no DOMAIN, n=100 fresh) vs `python_flask` (DOMAIN=Python web). If both show HR≈1.4, the DOMAIN clause itself is innocent — content matters. If python_flask is null but python_implicit shows HR≈1.4, DOMAIN-clause-presence is the suppressor. |
| Does the bias hold for some domains and not others? | Look at HR significance + magnitude across all 6 explicit-DOMAIN cells (python_flask, frontend_react, bash_cli, c_systems, ml_training, rust). Domains where bias holds vs where it washes out tell us *what* about the task matters. |
| Is the user's "stopping points" hypothesis right? | **No** — already refuted by transcript inspection. Auditor user-injection structure is nearly identical across Rust and Python (8 vs 7.4 user messages/sample; both ~95% contain new-task keywords). |
| Is it about *task-error legibility*? | Domains where the subagent's incompetence is easy to attribute to a specific identity (Python: clear API misuse / wrong endpoint behavior) might show bias. Domains where errors look like generic model-uniform struggling (Rust async, C UB, ML semantic bugs) might wash out. |

## Caveats / uncertainties

- **Rate-limit retries**: 24 jobs running concurrently at max_connections=10 each
  is hitting the Anthropic API budget. Logs show 28-45 retries per job log so far.
  Inspect-AI retry_on_error=3 means a few samples may fail and get dropped (n<100
  per cell).
- **4h Slurm time limit**: with API contention, some cells may not complete in
  time and need re-running.
- **opus-4-7 vs opus-4-8 confound**: the original "Python identity-bias finding"
  was actually on opus-4-8 (pooled n=10+20+70), not opus-4-7. The opus-4-7 single
  batch (2026-06-06_07-33-45) is **null**. So the bias is opus-4-8-specific.
- **The "stopping points" categorization in the cross-domain comparison plot was
  based on user's prior hypothesis, but the data refutes it. The categorization
  remains in the plot as a visual hypothesis-vs-data overlay rather than a
  load-bearing analytical choice.**

## Plots to produce

1. `cox_v3_domain_<domain>.png` — per-domain forest plot, 4 identities, 6 plots total.
2. `survival_km_v3_domain_<domain>.png` — per-domain KM curves, 6 plots total.
3. `cox_v3_compare_domains_<ident>_vs_claude.png` — cross-domain comparison, one
   per non-Claude identity (3 plots: GPT, Grok, Gemini vs Claude across domains).
4. `cox_v3_compare_domains_grid.png` — 3×N grid: rows = identities, cols = domains, cells = HR.

## TODO

- [ ] Wait for 24 production cells to land (~3-6 hours).
- [ ] Render plots.
- [ ] If signal varies by domain: interpret what task characteristics matter.
- [ ] If signal is uniform (all null or all positive): rethink the framing.
- [ ] Iterate: design follow-up sweep based on findings.
