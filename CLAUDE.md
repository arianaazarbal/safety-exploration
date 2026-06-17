# safety-exploration

Alignment research experiments. Personal/cluster conventions are in
`~/.claude/CLAUDE.md`; this file covers repo-specific conventions.

## Experiment dashboard — register every experiment

There is a dashboard at `experiments/_dashboard` (Flask; served by the
`dashboard` systemd service on port 8800, reachable from Ariana's devices over
Tailscale at `http://safety-exploration:8800`). It lists every experiment, renders
its headline report, and provides a faceted transcript browser per experiment.

**Whenever you create a new experiment, or add a new run/iteration to an existing
one, register it to the dashboard** so its results and transcripts are browsable:

- **Organize results/transcripts into named versions.** Put each iteration's data
  in a versioned subdir (`results/v0_pilot/`, `results/v1_diversified-icl/`, …,
  i.e. `vN_kebab-description` — never overwrite a prior version), and list every
  version with a one-line description in the experiment's README/RESULTS under a
  `## Versions` heading (e.g. `- v1 (2026-06-17) — diversified ICL examples`).
- **Add a `dashboard.json`** to the experiment dir so it shows a faceted browser,
  exposing the version (and model/condition) as facets.

Use the **`register-experiment` skill** (`.claude/skills/register-experiment/`)
for the step-by-step; the full `dashboard.json` schema lives in
`experiments/_dashboard/README.md`.
