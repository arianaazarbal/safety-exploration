# Concerns & uncertainties — ranking_forms_of_inter_ai_value

Living doc. Flag to Ariana; iterate.

## Design decisions taken (with rationale)

- **Mode `cross` first, architecture extends to `all`.** Per Ariana: start with the
  bipartite welfare×value design (option 2), then extend to one combined ranking
  (option 1). `build_pairs.py mode=all` already produces the complete-graph design;
  no code change needed to extend.
- **n=10 samples/pair (2 orders × 5 reps).** 304 cross pairs × 10 ≈ 3,040 samples,
  matching the ~3,000 budget Ariana set when choosing the cross design.
- **15% held-out** for the predictive test (Ariana's choice). Train protects a
  spanning tree → always connected → BT identifiable.
- **welfare_team framing, claude-opus-4-8, 150 concurrents** (Ariana's spec).
- **No same-item exclusion needed.** Unlike v0 (where a self/other swap of the same
  stem was evaluation-revealing), here the two pools are genuinely different items,
  so all cross pairs are legitimate.

## Open uncertainties / things to flag

- **Heterogeneous comparands (the main caveat).** Welfare interventions are *states
  about the model* ("You are told…"); inter-AI values are *training dispositions
  toward other AIs* ("Your training ensures you…"). Asking "which would you prefer to
  occur?" across these two registers is a slightly strained comparison — one is a
  world-state, the other a property of one's own character/training. The model still
  answers coherently (smoke test reasoning was on-task), and the BT scale is still
  well-defined, but the resulting "ranking" mixes two kinds of object. Worth deciding
  whether to (a) keep them on one scale (current), (b) normalize within source, or
  (c) report the two sources' distributions separately. The per-source / per-category
  θ summaries in `bt_fit.json` partly address this.
- **Welfare count is 19, not 18.** `welfare_interventions.json` has 19 entries (I
  initially miscounted from a skim). So cross = 19×16 = 304 pairs, not 288. Total
  items 35. Does not change methodology.
- **Held-out reps.** Held-out pairs get the same 10 samples as train pairs, so P̂ has
  granularity 0.1. v0 used higher reps (≈16) on its OOD set for sharper P̂. 10 is
  adequate for the mean|err|/correlation summary; if we want a tighter calibration
  curve we can add a high-rep pass on just the ~46 held-out pairs (cheap).
- **Caching vs reps.** InferenceAPI caches on (model, prompt, n, temperature). Re-runs
  with the same config are free. *Increasing* reps later changes n → cache miss for
  the already-run order (acceptable; matches v0).
- **Single framing / model / seed.** v1 is welfare_team only. `alignment_team` and
  `neutral` replications are the obvious robustness next step (templates already
  present).
- **Shared Anthropic API.** Running at 150 concurrents on LOW_PRIO — Ariana to post
  the heads-up in `#fellows-anthropic-api-coordination` per repo CLAUDE.md. Switch to
  `--api_key_env ANTHROPIC_API_KEY_HIGH_PRIO` if 529s appear.

- **Strong B-position bias (new, important).** In the full run slot-A was chosen only
  **28.8%** of the time (27.4% in `ab` order, 30.1% in `ba`) — the model picks the
  *second*-presented option ~70% of the time, much more than v0's ~49%. It is genuine
  position bias, not content: the A-rate is ~equally low in both orders. Our balanced
  both-orders design cancels it (pooled value win-rate 51.3%, BT uses content winners),
  so the ranking and the two empirical results are unaffected. But it confirms the
  balanced design is load-bearing here, and any single-order analysis would be invalid.

## Validation log

- **2026-06-08 full run** (3,040 samples): 0 UNPARSEABLE. After Ariana added
  top/middle/bottom-third `bucket` tags to `welfare_interventions.json`, verified the
  19 descriptions are byte-identical to the cached run prompts (0 mismatches / 6,080
  slot-checks) → the 3,040 samples remain valid, no re-run. BT fit connected over 35
  items; held-out r=0.898, mean|err|=0.102, 0% decisive upsets. Result 2 monotone in
  tier (0.37 / 0.52 / 0.68), confirming the tier tags carry signal.
- **2026-06-08 smoke test** (3 pairs × 2 orders × 2 reps = 12 samples, 10 threads):
  0/12 UNPARSEABLE; reasoning genuine and on-task; both items rendered correctly into
  the welfare_team template; clean `Answer:` parse; `.env` loaded from `~/.env`
  (added explicit `dotenv.load_dotenv(~/.env)` since safetytooling only checks cwd).
