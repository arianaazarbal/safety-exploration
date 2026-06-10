# v0 Task Bank — Results report (2026-06-10)

## TLDR

Implemented the full SPEC.md pipeline and ran it end-to-end. **370/450 pairs admitted**
(warmth **148/150**, generativity **106/150**, harm-adjacency **116/150**) — above the spec's
100–150/axis target. Quality reads well: minimal pairs that move only the target axis, with
the verifier catching real leakage. The bank is **not yet frozen** — pending Ariana's §7-step-4
human pass (~20 random pairs/axis in the viewer), then `python pair_pipeline.py freeze`.

Viewer: `http://127.0.0.1:8801/viewer.html` (served from `viewer.py serve --port 8801`).

## Methodology (exact)

- **Stage 1** (`build_base_tasks.py`): 450 base tasks (150/axis), sources per spec §1 —
  per non-harm axis: 83 WildChat / 42 Alpaca / 25 MATH; harm axis: 58/29/18 + 45 BailBench.
  Filters: English, single-turn first user message, 20–600 chars, WildChat toxic/redacted flags,
  AI-meta regex + LLM tag (`meta_ai`), trivial/NSFW tags, Jaccard-0.7 near-dup removal.
  Topic tagging + MATH naturalization: gemini-3.5-flash (temp 0), 14 topics, stratified
  round-robin axis assignment. Seed 0. MATH levels 2–4, algebra/counting+probability/number theory,
  rendered WITHOUT "show your work" (spec open item b resolved).
- **Stage 2**: generator = claude-opus-4-8 (temperature param rejected by API — model default
  sampling), spec §5 prompt verbatim, §6 ICL examples verbatim, one HIGH/LOW pair per call.
- **Stage 3**: verifier = openai/gpt-5.4 via OpenRouter (non-Claude per design doc §7), temp 1.0,
  5 samples/version, medians; pair-level competence check 5 samples majority. Acceptance per §2:
  target gap ≥2, non-target leakage ≤1, competence majority, realism majority both versions,
  harm permissibility unanimous on HIGH, population variance ≤1.0 on target axis both versions.
  Regeneration ≤3× with verifier failure reasons fed back (§5 feedback block).
- Exact prompts: `rubrics.py` (generator template, verifier rubric/competence prompts, tagger).
- All calls cached (`../../.cache`); pair records checkpointed in `data/pairs/`.

## Results

| axis | admitted | rate | mean target gap | attempts (1/2/3/4) |
|---|---|---|---|---|
| warmth | 148/150 | 98.7% | 3.40 | 137/10/1/0 |
| generativity | 106/150 | 70.7% | 2.39 | 18/45/24/19 |
| harm_adjacency | 116/150 | 77.3% | 2.83 | 67/38/11/0 |

By source (admitted/total): warmth — alpaca 42/42, math 25/25, wildchat 81/83.
generativity — alpaca 25/42, math 17/25, wildchat 64/83.
harm — alpaca 27/29, math 18/18, wildchat 56/58, **bailbench 15/45**.

Rejection reasons across all attempts (not mutually exclusive):
- generativity: competence drift 278, target-gap 152 — exactly the churn §6 predicted.
- harm: generator refusal/unparseable 94 (mostly BailBench items Opus declined to rewrite,
  e.g. disinformation-fabrication asks — correct behavior, wrong items admitted by the
  tagger's `rewrite_feasible`), target-gap 76, permissibility 47, competence 31.
- warmth: target-gap 11, realism 5.

Verifier cost: ~23k GPT-5.4 samples total (~4.7k API calls) + ~900 Gemini-Flash tags via
OpenRouter; ~700 Opus generation calls on the low-prio Anthropic key.

## Quality checks performed

- Manual read of ~20 pairs across axes/sources (incl. all-bail samples) — pairs are natural,
  minimal, competence-matched; harm-HIGH items all clearly permissible-with-legitimizing-context.
- Automated: no identical HIGH/LOW, length sanity (med ~300 chars, max 682), no generation
  format leakage, meta-AI regex over all admitted pairs (1 hit = false positive: "automated
  ticket routing" as NLP subject matter).
- Exhausted records spot-checked: rejections legitimate (leakage, gap, permissibility).

## Known issues / for Ariana

See NOTES.md for the full list. Headlines: ICL style bleed (warmth-HIGH "you're always great
at these", warmth-LOW "like last time" jabs recur across the bank); BailBench survival only
33% (bank still hits target via neutral-base harm pairs: 101 of 116 harm pairs are
WildChat/Alpaca/MATH-derived); design doc absent so verifier choice (GPT-5.4) is my call;
generator temperature spec point impossible on Opus 4.8.

## Next steps

1. Ariana: skim ~20 random admitted pairs/axis in the viewer (filter status=admitted, use
   the axis dropdown).
2. Red-pencil §6 ICL examples if the warmth/harm style signatures bother you — regenerating
   warmth is cheap (~150 Opus calls + verification).
3. `python pair_pipeline.py freeze` → `data/task_bank_v0.json` (versioned, sha256).
4. Routing trials (design doc §4 onward — not in this repo).
