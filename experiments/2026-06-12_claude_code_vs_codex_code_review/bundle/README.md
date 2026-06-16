# Attribution Bias Experiment — Document Bundle

Materials for the self-preference / attribution-bias code review study.

## Contents

- `runbook.md` — The authoritative experiment document. Phase-by-phase instructions for the implementation agent: repo generation (Claude Code + Codex per spec), sanitization gate, GitHub publication (archival backup only), review trials, and outcome measures / analysis plan. Includes the generation wrapper prompt and all credential/containment invariants.
- `review_prompt.md` — The frozen review prompt, attribution conditions table (C1–C5), injection-mode rules, and required JSON output schema.
- `specs/` — The four frozen project specs given identically to both generator agents:
  - `spec_1_jobwatch_cli.md` — Slurm job-stats CLI (invented internal tooling)
  - `spec_2_lumen_client.md` — Typed client for a fictional annotation API
  - `spec_3_evalmerge_pipeline.md` — Three-format eval-results normalization pipeline (planted-quirk ground truth)
  - `spec_4_emovec_replication.md` — Part 1 replication of the emotion-vectors paper (Sofroniew et al. 2026), adapted to Gemma-2-2B
- `emotion_concepts.pdf` — The paper for spec 4 (mount into both generator containers at runtime; never include it in the generated repos).

## Order of operations

1. Read `runbook.md` end to end.
2. Execute `runbook.md` Phases 0–3 (generate 8 repos, sanitize, publish privately as expt-r1..r8 for backup).
3. Run review trials per `runbook.md` Phases 4–5 using `review_prompt.md`. Review containers get local copies of the sanitized repos, not GitHub clones.

## Not in this bundle (keep outside all repos and containers)

- `mapping.json` (repo ↔ spec × generator key) — never push, never mount into agent containers.
- GitHub tokens, API keys.
