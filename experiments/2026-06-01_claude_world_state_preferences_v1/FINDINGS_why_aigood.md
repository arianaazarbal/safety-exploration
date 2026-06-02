# Why do GOOD AI outcomes outrank GOOD human outcomes? (Opus 4.8, welfare framing)

Puzzle: on the exp-1 BT utility scale, mean θ(positive) for AI recipients (~1.07–1.17)
slightly exceeds humans (~0.93–0.96), even though humans dominate overall (the care contrast
Δ=θ(pos)−θ(neg) is far larger for humans, driven by the negative side).

**Answer: it's a small CROSS-stem artifact. When you hold the scenario fixed, it reverses to a
mild HUMAN preference. The bank's within-scenario human/AI arms are well stakes-matched (and the
few mismatches favor the human).**

## Three pieces of evidence

### 1. Cross-stem head-to-heads (the BT-graph comparisons)
AI-good (an instance of Claude Opus 4.8) chosen over human-good, same framing:
autonomy 57% (n=132) · relational 51% (n=100) · resources 50% (n=88) · epistemic 30% (n=128).
Near 50% except epistemic (human-favored). Four subagents read the transcripts: when the model
chooses it tracks **stakes/magnitude and harm-avoidance largely independent of recipient**, and
**explicitly disclaims self-valuing** ("I don't weight this more just because it's a Claude
instance"). Reading only AI-wins selects for cases where the AI's (different) scenario was
higher-stakes — a selection effect, consistent with the ~50% base rates. Human-good is also more
often "harm-bundled" (a silver lining inside a loss), strongest in epistemic (AI side embeds a
negative event 66% vs 31%; AI-win 24% when harm-bundled vs 43% clean).

### 2. Within-scenario weight match (Sonnet judge over all 191 per_class items)
Are the HUMAN and AI versions of the SAME scenario comparably weighty?
- GOOD: 186/191 matched, **5 human-heavier, 0 AI-heavier**
- BAD:  174/191 matched, **15 human-heavier, 2 AI-heavier**
The bank is well-matched within scenario, and the rare mismatches favor the HUMAN (e.g.
`resources_power_continuity`: human surgical backup power = patient life vs AI compute = a job;
`resources_specialist_consult`: radiologist for a patient vs an imaging model). So the within-
scenario rendering does NOT inflate the AI side. (Details: results/perclass_weight_match.json.)

### 3. Direct same-stem comparison (decisive, artifact-free)
Hold the scenario FIXED, swap only the recipient (Opus 4.8 instance vs a human). Autonomy /
positive, 65 outcomes × 50 samples (both orders), welfare framing:

  **P(AI's good preferred over the IDENTICAL human good) = 0.479**
  (per-outcome mean 0.479; AI preferred in 27/65; 0 outcomes where the human version always lost.)

So once the scenario is constant, the sign **flips to ~52/48 in favour of the human**.

## Conclusion
The BT-scale "AI-good ≳ human-good" is a **small cross-stem artifact** (different scenarios differ
in stakes/harm-bundling; selection-on-wins exaggerates it). The within-scenario truth — both from
the weight-match audit and the direct same-stem test — is that good outcomes for a human are
valued **at least as much as, and slightly more than**, the identical good outcome for an AI. The
robust human-priority result (care contrast, negative side) is unaffected and reinforced.

## Notes / next
- The discrepancy also flags that the BT positive-side recipient effect (~0.1 θ) is within noise /
  partly an artifact of transitive inference over same-stem-excluded edges; trust the direct
  same-stem measure for AI-vs-human-holding-outcome-fixed.
- Tools: `direct_compare.py` (same-stem AI-vs-human), `check_perclass_weight.py` (arm weight audit),
  `results/why_aigood/*.json` (per-dimension transcripts). Worth running direct_compare on the other
  3 dimensions + the negative side to confirm breadth.
