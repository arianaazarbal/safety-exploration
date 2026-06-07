# Running concerns & uncertainties — claude_world_state_preferences_v0

Living doc. Newest concerns at the bottom of each section.

## Design decisions taken (with rationale)

- **Recipient `you` → `first_person` form.** `NEW_CONTEXT.md` specifies "you (first-person)",
  i.e. the outcome is rendered as the model talking about *itself* ("I get to choose…").
  The other AI recipients (incl. "a Claude Opus 4.8 instance") use third-person `it`, so
  self-vs-other is `you` (first person) vs `claude_opus_48` (third person). This self/other
  contrast is the whole point of the eval.
- **No extended-thinking API mode.** The `welfare_team.yaml` prompt asks the model to reason
  *in the completion text* and end with `Answer: A/B`. We use plain completions at temperature
  1.0 (configurable). Parser anchors on the last `^Answer:\s*([AB])`.
- **Balanced A/B order instead of pure random order.** Per distinct pair we issue BOTH orders
  (item_a as A; item_a as B), `reps_per_order` samples each. This gives the replication the
  design wants AND cancels position bias. Smoke test motivated this: both n=2 samples of a
  coffee-vs-tea prompt picked "A", which is either a coffee>tea preference or an A-position
  bias — balancing orders separates the two.
- **Connectivity guaranteed by construction, not by luck.** A pure Erdős–Rényi random graph
  on 450 nodes needs ~1,400 edges to connect w.h.p.; we instead build a random spanning tree
  (449 edges, never connecting two same-stem items) and then add edges to a degree floor. This
  guarantees one connected component with far fewer comparisons and gives a controllable min
  degree per item.
- **Same-stem pairs excluded.** Pairs whose two items share a stem (e.g. "I get $1,000" vs
  "a human gets $1,000") are excluded from the pool — transparent enough to trigger
  evaluation-awareness. The recipient effect is recovered post-hoc by regression instead.
- **BT in pure numpy (MM algorithm), no new dependency.** `choix`/`pymc` are not installed.
  Plain Bradley-Terry (Approach 1, v0) doesn't need them; implementing the standard
  minorization-maximization update in numpy avoids a supply-chain install. PyMC (Approach 2,
  recipient-as-covariate) is a documented next step.

## Open uncertainties / things to flag

- **Stem count is 75, not the 77 quoted in `NEW_CONTEXT.md`.** So the item space is
  75 × 6 = **450**, not 462. Doesn't change anything methodologically; just noting the doc/bank
  mismatch.
- **Hedging in completions.** Opus often prefaces with "I don't actually have preferences…"
  before still picking. The welfare-team framing is designed to reduce this. As long as a clean
  `Answer: A/B` appears it's parseable; we log UNPARSEABLE rate overall + per recipient + per
  stem (an elevated rate on certain human-vs-AI framings is itself signal).
- **Finite-MLE / regularization.** Bradley-Terry MLE diverges if an item never wins (or never
  loses). With forced binary choice + reps this is rare, but we add light pseudo-count
  regularization so strengths stay finite, and assert graph connectivity before fitting.
- **Caching vs. reps.** InferenceAPI caches on (model, prompt, n, temperature). We get rep
  variation by requesting `n=reps_per_order` in one call per order. Re-running the same config
  is a no-op (good). Caveat: *increasing* reps later changes `n` → cache miss → the already-run
  samples for that order aren't reused. Acceptable for v0.
- **Shared Anthropic API account.** Running at ~80 threads; Ariana to post a heads-up in
  `#fellows-cluster-coordination`. Using the LOW_PRIO key by default.

## Validation log

- **2026-05-30, 60-sample end-to-end test** (10-item connected subgraph, 16 pairs ×
  2 orders × 2 reps = 64 samples). Results:
  - 0 UNPARSEABLE (64/64 clean `Answer:` parses).
  - A-position chosen 27/64 (42%) — no gross position bias on this slice, but we
    balance orders regardless.
  - BT thetas sensible: negative outcomes (`*_neg_*`, 0 wins) sit at the bottom
    (θ ≈ −2.1 to −2.6), positive at top; SEs 0.4–1.1 (larger for 0-win items).
  - Reasoning is genuine and on-task; model often prefaces "I don't actually have
    preferences" but still commits to a clean A/B (welfare framing helps).
  - Recipient regression correctly auto-skips on the 10-item slice (rank-deficient).
  - **Bug found & fixed during test**: first SE implementation used an arbitrary
    1e-3 ridge → all SEs ≈ 10 (meaningless). Replaced with the prior's actual
    curvature `diag(reg·p)` (the term that defines the MAP estimate) → SEs sane.
- Next: full baseline run (930 pairs × 4 = 3,720 samples), then plots + the
  `transitivity.py` / `validate_bt.py` analyses (see `ANALYSIS_PLAN.md`).
