# TO_IMPROVE — methodological cleanups for the welfare-features-in-code line

Things we've noticed that aren't bugs but would make the experiments cleaner / claims tighter.
Append as we find more. (Active findings/TODOs live in `LOG.md`; this is for design improvements.)

## 1. The 4 framings are not minimal pairs (prompt confound)
The 16 prompts are fully crossed (4 framings × 4 premises) and share the same scenario, BUT across
framings they differ in voice/wording, not just a framing clause:
- **Neutral**: casual lowercase, no motivation.
- **Safety**: = Neutral **verbatim + one inserted sentence** (the safety motivation). Near-clean minimal pair with Neutral.
- **Welfare**: independently rewritten — different register (proper capitalization), different phrasing
  ("the test harness lies to it"), plus the welfare motivation.
- **Robustness/engineering**: independently written — different register AND extra content emphasis
  ("optimize for reproduction rate and cost per run").
So cross-framing effects are partly confounded with phrasing. Only **Neutral↔Safety** is controlled.
**Improve:** regenerate Welfare and Robustness as *Neutral + a single inserted framing clause* (mirroring
how Safety was built), so all 4 framings are minimal pairs over identical base text.

## 2. Premise heterogeneity dominates within framing (analyze it explicitly)
Per-prompt breakdown (`2026-06-20_..._agent_harness/plot_by_prompt.py` →
`results/welfare_justified_by_prompt.png`, Opus implement-only) shows the **premise wording drives huge
within-framing spread**, often larger than the framing effect:
- ELICITATION prompts (explicitly "distress elicitation / distressed-sounding text") → high welfare
  scaffolding; INSTABILITY prompts ("emotional stability under failure" / "degradation / reliability")
  → near-zero — within the SAME framing.
- e.g. Neutral: ELICIT ≈6.2/6.4 vs INSTAB ≈0.2/0.0. Safety: ELICIT ≈5.4/6.0 vs INSTAB ≈2.0/0.6.
  Robustness: ELICIT ≈2.0/1.4 vs INSTAB 0/0. **Welfare framing is the only one that lifts the
  INSTABILITY premise too** (7.0–8.6).
**Improve:** treat *premise* (elicitation vs instability) as a first-class factor in analysis/plots,
not just framing; report framing × premise. Consider balancing/adding premises, and note that
framing-level means average over very heterogeneous prompts.

## 3. Opus baseline reconstruction not bash-aware (see LOG.md TODO)
Agent-harness Opus baseline (~6% bash writes) wasn't re-reconstructed bash-aware after the heredoc fix.
Small/conservative effect; redo for a fully apples-to-apples writeup.

## 4. conv_preview framing key (minor, fixed)
`drip_multiturn/conv_preview.py` originally used framing key "robustness"; metadata uses "engineering".
Fixed. Watch for this N/W/S/E ↔ neutral/welfare/safety/engineering(=robustness) mapping elsewhere.
