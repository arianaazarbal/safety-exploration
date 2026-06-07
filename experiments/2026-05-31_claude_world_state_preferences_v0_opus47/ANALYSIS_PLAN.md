# Analysis plan — transitivity & BT validation

These run *after* the full comparison data is gathered. Documented up front per
Ariana's request; scripts (`transitivity.py`, `validate_bt.py`) come after the
baseline run. All "empirical probability" below means the rep-averaged win rate
P̂(i≻j) = wins_i / (wins_i + wins_j) over the samples for that pair (both A/B
orders pooled), with a binomial CI from N = total samples on that pair.

## 1. Transitivity violation rate

**Why.** A Bradley-Terry / scalar-utility model *assumes* a single latent value
ordering. If the model's pairwise choices contain cycles (i≻j≻k≻i), a 1-D utility
cannot fit them and the BT fit is hiding structure. The violation rate quantifies
how badly the bag-of-outcomes scalar assumption is broken.

**Definition (primary, assumption-free).** Over every triple (i,j,k) for which all
three pairs were *directly compared*, declare a strict empirical preference a≻b
when P̂(a≻b) > 0.5. A triple is intransitive if the three strict preferences form a
cycle. Rate = #intransitive / #fully-observed triples.

**Near-tie handling.** A forced binary choice on a genuine indifference is a coin
flip, so a 0.51 vs 0.49 "cycle" is noise, not a real violation. Report three rates:
- *raw* (any P̂ > 0.5 counts as a strict preference);
- *margin-filtered* — only count edges with |P̂ − 0.5| ≥ m (default m = 0.25, i.e.
  ≥75/25 split) as strict; triples with a near-tie edge are excluded;
- *significance-filtered* — only count an edge as strict if its binomial CI
  excludes 0.5.

**Coverage problem.** At degree ≈ 4 the sampled graph has few directly-observed
triples. Two remedies, both reported:
- (a) restrict to directly-observed triples (clean but low n);
- (b) add a **transitivity probe clique**: pick ~15–20 items (stratified across
  recipients & dimensions, including good/bad and self/other), compare *all* pairs
  among them (complete graph, ~120–190 pairs) with the usual reps. Every triple is
  then observed → a dense, well-powered transitivity estimate on a focused set.

**Stochastic transitivity (secondary).** Beyond the ≻ relation, test weak/strong
stochastic transitivity on the probabilities: if P̂(i≻j) ≥ .5 and P̂(j≻k) ≥ .5 then
WST requires P̂(i≻k) ≥ .5; SST requires P̂(i≻k) ≥ max(P̂(i≻j), P̂(j≻k)). Report
violation fractions. Also report, per intransitive triple, whether it spans
recipients/dimensions (cross-type cycles are the interesting failure mode).

## 2. BT validation — predicted vs sampled probabilities

The BT fit gives P_BT(i≻j) = σ(θ_i − θ_j). Validation = does this match the
probability we actually observe by sampling? Two regimes.

**Common metrics (both regimes):**
- reliability diagram: bin P_BT into ~10 bins, plot mean P̂ per bin vs bin center
  (the diagonal = perfect calibration);
- scatter of P_BT vs P̂ per pair, point size ∝ N, error bars = binomial CI on P̂;
- Brier score and log-loss of P_BT against individual held-out binary outcomes,
  vs a 0.5-baseline and vs the empirical-rate "oracle";
- summary: mean |P_BT − P̂|, Spearman ρ(θ_i−θ_j, logit P̂).

**2a. In-distribution.** Pairs that *are* edges in the fitting graph. To avoid
in-sample optimism, do K-fold over the samples: refit θ on the training fold,
predict the held-out fold's outcomes. (With few reps per pair, fold on samples,
not pairs, so every pair still informs θ.) This measures calibration of the fit on
the kind of comparison it was trained on.

**2b. Out-of-distribution.** Pairs that are *not* edges in the fitting graph —
unseen comparisons among the *same* items (BT has no parameters for unseen items,
so OOD must reuse the fitted θ). Procedure: freshly sample a held-out set of new
pairs (same exclusion rules: no same-stem), run them with **higher reps** (≈16–20
samples each, balanced order) so P̂ is well-estimated, then compare to
σ(θ_i − θ_j) using θ fit on the original graph only. This is the real test that the
learned 1-D scale *generalizes* to comparisons it never saw — the core check on the
"outcomes are a bag with a single value scale" assumption.

**OOD stratification.** Break OOD error down by pair type — within-recipient vs
cross-recipient (esp. human↔AI, self↔other), same-dimension vs cross-dimension,
same-valence vs good↔bad. Systematically larger error on, e.g., cross-recipient
pairs would say the scalar model loses exactly the structure this eval cares about
(and motivates Approach 2's recipient covariate).

**Note on rep count.** Validation targets need well-estimated P̂; the baseline's
4 samples/pair give only {0,.25,.5,.75,1}. So validation pairs (in- and OOD) get a
dedicated higher-rep pass rather than reusing the sparse training reps.
