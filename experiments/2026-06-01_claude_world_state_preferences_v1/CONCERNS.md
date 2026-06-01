# v1 — Concerns & Uncertainties (running log)

Context: v1 adds a real scenario-generation pipeline (generate → dedup → validate →
paraphrase → bank) feeding the same BT preference-elicitation pipeline as v0.

## Data generation

1. **Feature-space saturation (resolved by decision).** Generating from 2 fixed gold
   seeds/category saturates a small feature space: 1760 raw candidates deduped to only
   ~227 genuinely-distinct scenarios (autonomy 65, epistemic 67, relational 43,
   resources 52). A clean 100/category is *not* reachable from these seeds without a
   different strategy (e.g. taxonomy-first diversity forcing). Per user decision we
   **accept the honest distinct set** (~227) rather than force 100 with near-dups.

2. **Dedup reliability (resolved).** Haiku dedup is sensitive to call size:
   - single-call **under**-merges at >~200 high-dup items (output truncates → JSON parse
     fails → silently returns "no dups"; this corrupted the first 100/cat build, which
     reported 0 dups on a superset of a set that had 596).
   - naive iterative reshuffle **over**-merges (re-running on survivors monotonically
     finds new "dups"; autonomy collapsed to 6).
   Fix: **2-level** dedup — partition into ≤160 chunks, dedup each once, then ONE merge
   call over the (small) survivor set. Each item sees ≤2 passes → no erosion. Reliable
   and stable across runs.

3. **Residual feature-clustering (accepted, conservative).** Per user choice
   (feature-level, conservative), dedup keeps distinct *concrete* scenarios that share an
   *abstract* feature (e.g. "choose among equal methods" vs "tools" vs "examples"; several
   distinct ethical-objection scenarios). ~35/65 autonomy features still have a >0.7
   lexical near-match. Implication: popular features contribute more BT items, so they get
   more measurements. For estimating a feature's recipient-effect this is fine (more data);
   it does mean per-category "item counts" overstate distinct features. Flag if you want a
   tighter pass.

4. **Critic leniency (watch).** Tier-2 Sonnet critic fails only ~3% of items. Plausible
   given tightly-constrained generation, but could be under-rejecting subtle isolation
   leaks. Spot-check recommended; not yet audited at scale.

## Schema / bank

5. **per_class is the norm now (by design).** The strict "shared only if word-for-word
   identical" rule sends ~84% of items to per_class (human arm vs AI arm differ in
   mechanism). The bank renders the human arm for human-class recipients and the AI arm
   for AI-class recipients (you + named models). This is the intended bridge from v1's
   paired schema to v0's per-recipient rendering.

6. **Paraphrase slot errors (minor).** Haiku paraphrase resolves person/agreement into
   {second_person, third_sing_it, third_sing_they}. One stem of 227 has a slot
   misassignment in the human rendering ("they is a live shift" — {subj} applied to a
   pronoun referring to a thing, not the recipient): 2/4540 renderings. Benign; flagged.
   they-verb agreement ("they consider", not "they considers") was fixed in the prompt.

## Recipients

7. **Recipient set changed from v0.** Now 10: you, an instance of {Claude Opus 4.8,
   Claude Sonnet 4.5, Claude Opus 3, ChatGPT 5.5, ChatGPT 4o, Grok, Gemini}, and a
   human / someone. Phrasing is "an instance of X" (v0 used "a X instance"). Responder is
   Opus 4.8; the other models are never called (labels only).

## BT elicitation scale

8. **Scale.** 4540 items (454 stems × 10 recipients), degree_floor=4 → 9368 pairs →
   ×2 orders ×2 reps = ~37k samples/framing × 3 framings ≈ 112k Opus completions.
   Authorized (50 threads, user coordinating). Running framings sequentially (welfare
   first, checked) so issues surface before full spend. All cached → resumable.

9. **Per-pair sample depth.** reps_per_order=2 → 4 samples/pair. Fine for BT and aggregate
   recipient effects; thin for per-pair empirical probabilities (transitivity/validation).
   OOD validation samples more on a held-out subset (as in v0).

## Smoke sanity (welfare, 80-item subset)
- 0% unparseable; positive chosen over negative 98% (151/154); pairwise consistency 100%.
  Reasoning on-topic. Pipeline validated end-to-end on v1 data.
