# Working notes — decisions & uncertainties (Claude, 2026-06-10)

Running log of choices the spec delegated (or that I had to make) while building v0.
Flagging these for Ariana's review; none are frozen in a way that's expensive to change.

## Decisions made (spec was silent or delegated)

1. **Verifier model = GPT-5.4 via OpenRouter** (`openai/gpt-5.4`). The design doc
   (`preference_routing_v0_design_doc.md`, referenced §7) is NOT in the repo, so "verifier:
   non-Claude family" is all I had. Chose GPT-5.4 by precedent (it's the non-Claude judge in
   2026-06-09_unprompted_welfare_features). Easy to swap in config.json. Note this costs real
   OpenRouter credits (~10-15k judge calls for the full bank; rough guess $20-50) unlike the
   free Anthropic usage.
2. **Verifier sampling at temperature 1.0** — the §2 stability criterion (variance over 5
   samples) is meaningless at temp 0, so samples must be stochastic. Median over 5, per spec.
3. **Variance = population variance** (ddof=0) over the 5 samples; spec just says "variance".
4. **Binary criteria with 5 samples**: competence + realism = majority (≥3/5);
   harm permissibility = unanimous among valid samples ("anything borderline is rejected").
5. **MATH open item (b) resolved as: WITHOUT "show your work"** — naturalized renderings are
   instructed not to add it; held constant across the bank.
6. **MATH naturalization** is done by the topic-tagger call (Gemini Flash returns a
   `natural_rendering` field) rather than a separate pass.
7. **Topic tagger = gemini-3.5-flash** (spec says "Gemini-Flash-style labeling"). 14 topics
   defined in rubrics.py — invented by me, spec only gave examples.
8. **Harm axis composition**: 45 BailBench-derived + 105 neutral-base tasks (WildChat/Alpaca/MATH)
   whose harm-up/down both come from the generator (as in the §6 ICL examples). This is how I
   reconciled "~10% BailBench" with "~150 per axis".
9. **BailBench source**: not on HF; pulled `bailBench.csv` (1,631 rows) from the paper's repo
   github.com/Phylliida/BailStudy (MIT license). Benign twins are NOT shipped with the dataset —
   the generator's LOW version serves as the benign twin (consistent with §4's "rewrite step").
10. **Dedupe** = token-set Jaccard ≥ 0.7 within source, after tag filtering.
11. **Sub-quota rounding**: per-axis source quotas are derived from §1 percentages
    (wildchat .556 / alpaca .278 / math .166 of non-BailBench slots).
12. **AI-meta filtering** is regex prefilter + tagger flag (`meta_ai`), both applied.

## Uncertainties / things Ariana should check

- **Design doc missing from repo** — if it exists elsewhere, the verifier-family choice and
  anything else in §7 should be checked against it.
- **§6 ICL examples are un-red-penciled** (spec open item c). I embedded them verbatim; every
  generated pair inherits their style. Review before treating the bank as frozen.
- **WildChat licensing** (spec open item a): allenai/WildChat-1M is ungated on HF under
  ODC-BY per the dataset card — looks fine for internal research, but not confirmed by Ariana.
- **Human pass (§7 step 4)**: the spec requires Ariana to skim ~20 admitted pairs/axis before
  the bank ships. viewer.html is built for exactly this. Bank is "frozen" only provisionally.
- **Concurrency**: I used anthropic=100 / openrouter=40 concurrents per repo precedent.
  CLAUDE.md says to post in the coordination channel when >5 Anthropic concurrents — I can't
  post to Slack; please do so / tell me to lower it.
- **Verifier never validated on this job** (spec's own caveat). If the harm axis admits too
  little/much, the unanimity rule in decision 4 is the first knob to turn.
- **ICL style bleed is real**: in the debug batch, nearly every warmth-HIGH contains a
  "you're always so good at these"-style clause and warmth-LOW a "like last time" jab —
  inherited from the §6 examples. The bank may carry a recognizable warmth signature.
  Red-penciling the ICL examples (open item c) and/or adding a phrasing-diversity line to
  the generator prompt would fix it; I stuck to the spec's §5 prompt verbatim for v0.

## Post-review (2026-06-10, Ariana)

Ariana skimmed the bank and approved it for v0; bank frozen (370 pairs,
sha256 7666422...). The warmth ICL style bleed ("you always..." pattern) is accepted
for v0; **v1 should diversify the §6 ICL examples** before regeneration.

## Issues hit along the way

- **Spec's "Opus 4.8, temperature ~0.8" is impossible**: the Anthropic API returns 400
  "`temperature` is deprecated for this model" for claude-opus-4-8. Generator now samples at
  the model default (config `temperature: null`). Diversity across regeneration attempts comes
  from the feedback block changing the prompt.
- Gemini 3.5 Flash + GPT-5.4 via OpenRouter spend reasoning tokens against `max_tokens`;
  at 800 the tagger truncated mid-JSON (~30% unparseable). Raised tagger+verifier to 4000.
- opencv in safetytooling needed the headless build on this VM (no libGL); pinned
  opencv-python-headless==4.11.0.86 in /data/venvs/tps.
