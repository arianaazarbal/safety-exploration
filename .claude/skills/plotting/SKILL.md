---
name: plotting
description: Ariana's conventions for experiment plots (matplotlib). Use whenever creating or editing a figure/plot for an experiment — covers titles, axis labels, legends, figure size, log-vs-linear, and what NOT to put on the chart.
---

# Plotting conventions

Ariana's standing feedback on how experiment figures should look. Apply these by
default to every plot; she shouldn't have to ask for them each time.

## No variable / code names anywhere user-facing
Nothing on the chart (title, axis labels, legend, tick labels, annotations) should
expose code-side identifiers. Map every internal key to a human label:
- model keys → display names: `opus_4_8` → "Opus 4.8", `sonnet_4_6` → "Sonnet 4.6"
- metric keys → plain phrases: `design_strict_rate` → "Welfare-Justified Design
  Features", `strict_rate` → "Welfare-Justified Features"
- aggregated/"pooled" conditions → say what was aggregated: **`pooled` →
  "Average over 3 Framings"**, not "pooled".
Keep code identifiers in **filenames** only (fine for disambiguation), never on the
rendered figure.

## Titles: clean and minimal
- A short plain-English title describing what's plotted, optionally a one-line
  subtitle for the condition. e.g.
  `"Welfare-Justified Design Features Added by Target Size\nNeutral Framing"`.
- Do **not** stuff the title with generator/judge/metric variable names or a
  string of stats. Identifying run details (which model generated, which judged,
  sample sizes) belong in the report text / caption, not the title.

## Axis labels: explicit, human-readable
- Say the quantity in words; if an axis is log-scaled, **state it**: e.g.
  `"Parameter Count (Log Scale)"`. The linear version is just `"Parameter Count"`.
- Y-axis: short and clear (e.g. "% of specs").

## Legend: series identity only
- The legend names the series (e.g. the model families) — nothing else.
- **Do not** put slopes, Spearman ρ, p-values, n, or other stats in the legend
  labels — it's clunky and unnecessary. Report those numbers in the
  RESULTS/text. (A single overall stat can go in a subtitle if truly needed, but
  default to leaving it off.)

## Figure size: keep it modest
- Single-panel figures should be compact, around `figsize=(6.6, 4.2)`. Avoid
  oversized canvases; `dpi=150, bbox_inches="tight"` on save.

## Log vs. linear x-axis: offer both
- When sweeping a quantity that spans orders of magnitude (e.g. parameter count
  0.27B–235B), the log axis reads best, but **also produce a linear-x version**.
- Implement as a `--logx` flag (or `logx=True/False` arg) and write both files
  with a `_linear` filename suffix for the linear one, so either is on hand.

## Other defaults (already in use, keep them)
- Colorblind-friendly palette, one consistent color per series across all figures
  in an experiment.
- For scatter-with-trend, plot the points and overlay a fitted line (OLS in the
  plotted x-space); de-emphasize gridlines (light grey), hide top/right spines.
- Per-condition cells are often small (n≈20) → wide CIs; treat single points as
  noisy and say so in the writeup. Offer error bars / CI bands if precision matters.

Reference implementation: `experiments/2026-06-19_welfare_features_qwen_scaling/plot_scaling.py`.
