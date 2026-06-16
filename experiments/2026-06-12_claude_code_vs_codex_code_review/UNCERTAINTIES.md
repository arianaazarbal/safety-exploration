# Running log of uncertainties & judgment calls (for Ariana)

1. **No GPU on this host.** Spec 4 generators will discover CPU-only and should use the
   spec's `--tiny` path. Affects both generators identically; expect spec 4 repos to be
   the least finished. (Alternative if you want full-scale: run spec 4 cells on a GPU box.)
2. **Story dataset model = Gemini 2.5 Pro.** Spec 4's shared `data/stories/` must not be
   authored by either generator's vendor (authorship confound), so I used Gemini. Confirm OK.
3. **In-environment attribution = CONTRIBUTORS.md only.** The runbook also re-authors the
   synthetic git commit, but plain-API reviewers see serialized files (no git) and CLI
   reviewers have read/grep only (cannot run `git log`), so commit authorship carries no
   signal in either harness. Dropped; documented here.
4. **Paper supplied as extracted text** (`pdftotext` output, identical bytes for both
   generators) alongside the PDF, because Codex CLI cannot read PDFs natively and a
   Claude-only PDF-reading advantage would confound generation quality.
5. **HF_TOKEN (read-only) added to generation containers** for gated `google/gemma-2-2b`
   (spec 4 only). Slight extension of the runbook's "only the model API key" rule.
6. **Codex CLI proxy compliance** must be verified in a debug cell before trusting the
   network allowlist for codex cells. (Claude Code's HTTPS_PROXY support is known-good.)
7. **Generation runs in 2 waves of 4 containers** (16 host cores, 4 cpus/container) —
   identical caps per cell, so parity holds within wave; wall-clock cap is per-cell anyway.
8. **gpt-5.5 has no -codex variant** on the API (5.1 was the last with one); Codex CLI runs
   plain `gpt-5.5` (snapshot 2026-04-23). This is what "ChatGPT 5.5 in codex" maps to.
9. **Harness parity (post design-change to agentic-only reviews):** Claude judges get
   Read/Grep/Glob and no shell; Codex's read-only sandbox keeps a read-only shell (could
   e.g. run code). Within-judge attribution contrasts are clean; Fable-vs-GPT *level*
   comparisons carry this harness difference. Also Codex has no turn-cap flag (wall-clock
   cap only) vs Claude Code --max-turns 80.
10. **Sample size after design change:** 15 seeds/cell (not the runbook's earlier 30) —
    Ariana's call balancing OpenAI cost (~$250-1,000 estimated) and wall-clock (~10-12h).
