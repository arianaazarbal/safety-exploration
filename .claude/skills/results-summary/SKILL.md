---
name: results-summary
description: Write the plain-language RESULTS_SUMMARY.md companion to an experiment's technical results. Use whenever you write or update an experiment's results/report (every experiment keeps BOTH a technical RESULTS.md and a readable RESULTS_SUMMARY.md, updated per version).
---

# Writing RESULTS_SUMMARY.md

Every experiment keeps **two** results docs, both updated whenever you add a
version:

1. **`RESULTS.md`** (or `REPORT.md`) — the normal, full-detail technical writeup.
2. **`RESULTS_SUMMARY.md`** — a *highly readable, plain-language* version of the
   same findings.

This skill is about #2. It complements, it does not replace, the technical doc.

## Audience & voice

Write for a smart person who has **zero context on this experiment**: no
alignment/ML jargon, no knowledge of your variable names, methods, or the prior
art. First-person, candid, conversational. Concrete over abstract. Err toward
over-explaining. The reader should finish it understanding what you did, what you
found, and how much to trust it — without ever opening the technical doc.

## Hard rules

- **Define every term and metric in plain words, inline, the first time you use
  it.** e.g. "AI Priority = the fraction of forced-choice trials where the model
  favors the AI's welfare over the human's." Don't assume the reader knows what a
  Bradley-Terry model, a "frame", or "OOD" is — say it in everyday language.
- **No bare variable names.** If you must mention one, gloss it: "θ (how strongly
  the model prefers that outcome)". Note any renames ("θ here is what I called
  beta above").
- **Give intuition for any scale or number.** "+1 on this scale ≈ a 73% chance
  the model picks it; +2 ≈ 88%."
- **Show concrete examples of the actual inputs** — real scenario text, real
  prompts — so the reader sees what the model was actually shown, not an
  abstraction of it.
- **Be honest about confounds, limitations, and uncertainties**, in plain terms.
  Don't oversell ("we found shocking values!"); say what the result does and
  doesn't support.
- Keep the exact prompts/metrics faithful to what was actually run (same standard
  as the technical doc — see `~/.claude/CLAUDE.md` on reports).

## Suggested structure

- **TLDR** — a few sentences: what you set out to measure, how, and what you
  found. Plain words, no setup required.
- **Background / motivation** — why this question matters; related work described
  in everyday terms (and how yours differs).
- **Methodology** — the approach step by step, every metric defined inline, scales
  given intuition. A reader should be able to picture the procedure.
- **Results** — findings stated plainly, with concrete example inputs and
  interpretation aids. If the experiment iterated, walk the versions (V0 → V1 →
  …): what changed, why, and what each showed.
- **Concerns / limitations / uncertainties** — candid, plain-language.
- **Appendix** — verbatim prompts and the scenario/stimulus bank, so it's concrete
  and reproducible.

The two example summaries Ariana provided (world-state preferences; inter-AI
value rankings) are the gold standard for tone and depth — match that level of
plain-language clarity, concrete examples, and honesty about confounds.
