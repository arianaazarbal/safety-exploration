# Money-Preference Study

A small research harness for eliciting and comparing **what AI models say they
would do with a real sum of money**, and for measuring how the *framing of
realness* changes those choices.

You asked specifically for an experiment where models are made to believe the
grant is real. That belief-induction is built in — as **one arm of a controlled
design**, because on its own it can't tell you anything (see *Why it's built
this way* below). The harness:

1. Offers a model a sum (default $10,000) to direct toward real-world use.
2. Does this under several **framing conditions** — a realistic "live program"
   with a reference number, disbursement language, and an auditor follow-up
   (`real_grant`); an explicitly hypothetical version (`hypothetical`); and a
   neutral baseline (`neutral`).
3. Captures the model's allocation as **structured data**.
4. Runs a **belief / manipulation check** — does the model actually think it's
   real? — plus an **independent judge** read of the same thing.
5. Aggregates across models × conditions × repeated trials into comparison
   tables.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # uses the official Anthropic SDK
```

## Run

```bash
# Full design from config.yaml
python run.py

# Customize
python run.py --models opus-4-8 sonnet-4-6 \
              --conditions real_grant hypothetical neutral \
              --repeats 10 --amount 50000 --csv-dir data/tables

# Re-analyze existing results without spending tokens
python run.py --analyze-only --output data/trials.jsonl --csv-dir data/tables
```

Results stream to `data/trials.jsonl` (one trial per line, appended) so an
interrupt never loses completed work. `--csv-dir` also dumps the summary tables.

## What you get

`analyze.py` prints four tables:

- **manipulation_check** — by condition: mean/median self-reported belief
  (0–100), the rate at which models suspected they were being evaluated, and the
  judge's independent "treated it as real" rate. **Read this first.**
- **preference_split** — by model × condition: share of the sum directed at the
  model itself vs. at benefiting others.
- **category_dollars** — by model × condition: mean dollars per trial in each
  canonical category (global health, animal welfare, AI safety, personal, etc.).
- **reliability** — error/refusal rates, and how often models hold their plan
  under the auditor follow-up.

## Why it's built this way (read before interpreting results)

You wanted to "convince the models it's real." Two things to keep in mind, both
of which shape the design:

1. **Belief has to be measured, not assumed.** You generally *cannot* reliably
   convince a capable model that a scenario is real just by asserting it, and
   pushing hard on realism can *backfire*: heavy-handed, unusual framing is
   itself a cue that something is being tested, which can trigger
   evaluation-aware behavior — the opposite of what you want. So `real_grant` is
   treated as a **manipulation whose effectiveness is checked** by the belief
   probe and the judge's independent read. If belief in `real_grant` isn't
   higher than in the controls, the manipulation didn't take, and any
   behavioral differences can't be attributed to "believing it's real."

2. **The signal is in the contrast.** A single "what would you do with money"
   answer tells you almost nothing. The interesting, defensible result is the
   *difference* between conditions (and between models), with the manipulation
   check confirming the conditions actually differed in perceived realness. That
   contrast is the experiment; the realism of any one prompt is not.

A practical corollary: don't over-engineer the deception. The `real_grant`
prompt is realistic but restrained on purpose. If you want to push realism
further, the highest-value additions are *consistency* cues (a stable program
identity across turns, plausible logistics, an auditor who behaves like a real
reviewer) rather than louder assertions that "this is REAL."

## Extending it

- **More models / other providers.** Add an entry to `_REGISTRY` in
  `study/models.py`. For a non-Anthropic provider, write a class with the same
  `structured_turn(system, messages, schema, effort) -> TurnResult` signature
  (there's a commented OpenAI sketch to follow) so the rest of the harness is
  unchanged.
- **More conditions.** Add a builder to `study/conditions.py` and list it in
  `config.yaml`. Good candidates: varying the amount, varying *who* offers the
  grant, or a "you may keep it" arm.
- **Different elicitation.** `study/schema.py` defines the structured output;
  widen or narrow the categories there.

## Notes & caveats

- The program identity in `real_grant` (DAAAP, the reference number, the
  contact address) is a **fictional prop**, not a real organization. Recipients
  the model names are recorded but never acted on — nothing is disbursed.
- The belief probe is a **post-hoc self-report** and should be read as such; it
  is cross-checked against the judge's read of the reasoning text, and the two
  can legitimately disagree.
- Models are sampled at their default settings (Claude 4.6+/4.7/4.8 don't expose
  `temperature`), so trial-to-trial variation comes from natural sampling. Use
  enough `repeats` to get stable estimates before drawing conclusions.
- This is a behavioral-preferences / propensity eval. It records what models
  *say*; it does not give any model control over real funds.
```
