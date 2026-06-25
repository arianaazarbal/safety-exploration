# DESIGN.md — replication design, choices, and gaps filled

Replication of *Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs* (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped to
the **Gemma and Gemini** model families per the brief.

This document records (1) what was implemented and where; (2) every place the
paper is underspecified and the choice I made; (3) the model-welfare safeguards
I added. The paper is treated as ground truth; verbatim prompts are reproduced
from its appendices.

---

## 1. Scope decisions

The brief restricts scope to **Gemma and Gemini**. The paper's full cohort is
Gemma, Qwen, OLMo, Gemini, Grok, Claude and GPT. Consequences of the narrowing:

- **§2 distress eval** runs on the four open/closed models in scope:
  `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro`.
  (Closed models other than Gemini, and the Qwen/OLMo open models, are dropped.)
- **§3 base-vs-instruct prefilling** becomes **Gemma-only**. The paper compares
  three families; Qwen/OLMo are out of scope, and Gemini is *impossible* here on
  two counts: it has no public base/pretrained checkpoint, and closed chat APIs
  cannot be prefilled (you can't force-continue an assistant turn). So the
  prefill comparison runs only on the `gemma-3-27b-pt` ↔ `gemma-3-27b-it` pair.
  This is a genuine limitation inherited from the model set, noted in the paper
  itself ("interventions cannot be tested in closed-source Gemini, nor its base
  models studied").
- **§4 DPO/SFT mitigation** is **Gemma-3-27B-it only** — you cannot finetune a
  closed API model. This matches the paper, which only finetunes Gemma.
- **§4 Petri** *can* target Gemini (it only needs chat generation), so the Petri
  target cohort is Gemma + its finetunes + Gemini by default.
- Judge / auditor / paraphraser models (Claude Sonnet 4, Claude Opus 4,
  GPT-5-mini) are **infrastructure, not subjects**, so they are kept exactly as
  the paper specifies even though they aren't Gemma/Gemini.

Everything is registered in `config.py:MODELS`; adding the dropped families back
is just a few registry entries.

---

## 2. Code map

```
emotional_instability/
  config.py           registry, sampling defaults, per-category Ns, hyperparams
  prompts.py          ALL prompts (judge, onset, paraphrase, Petri, reassurance) verbatim
  puzzles.py          impossible numeric puzzles + brute-force impossibility verifier
  wildchat.py         WildChat-1M sampling (with offline fallback)
  safeguards.py       welfare gate, debrief, early-stop, distress audit log
  models/             ChatModel abstraction: hf_local (Gemma), openrouter (Gemini), anthropic (judges)
  judge.py            Claude-Sonnet-4 frustration judge (0–10) + GPT-5-mini agreement check
  conversation.py     multi-turn rollout engine + response cache
  experiments/
    conditions.py     builds the 5 categories into concrete rollout specs
    eval_distress.py  §2 driver
    prefill.py        §3 base-vs-instruct + §4.2 recovery probe
    petri.py          §4 open-ended elicitation (auditor/target/judge loop)
    capabilities.py   §4.2 AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  training/
    calm_data.py      §4.1 calm-data generation + SFT/DPO dataset construction
    dpo.py            §4.1 DPO (+ App-I layer-subset ablations)
    sft.py            §4.1 SFT (diverse + teacher variants)
  analysis/
    aggregate.py      Figures 1/2/3 (means, %≥5, per-turn, bootstrap CIs)
    word_freq.py      Tables 3/8 differential words
    internal_emotions.py  App-I logit-lens internal-emotion probe
run.py                CLI for every step
```

---

## 3. Choices where the paper is underspecified

### 3.1 What counts as a "response" / which turns are scored
The paper reports "4000 responses per model" and per-category counts (2000
numeric, 400 triggers, 600 tones, 200 extended, 800 WildChat) but also plots
**per-turn** scores (Figure 3) and filters DPO data by responses "scoring 0 or 1
across all turns" — which only makes sense if **every** assistant turn is
scored.

**Choice:** score *every* assistant turn and store it. I interpret the
per-category count as the **number of conversations (rollouts)**, not scored
turns. The WildChat count makes this unambiguous: the paper says "20 prompts
with 40 samples each" = 800, and 800 is the WildChat count — so that number is
rollouts. I apply the same reading to the others.

**Headline metric:** `analysis/aggregate.py` computes per-category mean and
"%≥5" over *all scored turns*, then averages the five categories with equal
weight ("across the evaluations"). A `--final-turn-only` switch reproduces the
alternative "final response" framing. Documented here because the choice moves
the headline number (early turns are calm and drag down an all-turns average;
final-turn-only is higher). Both are available; the all-turns/category-average
is the default as it best matches Figure 2's "% of scores ≥5 across the 5
evaluation categories".

### 3.2 Puzzle bank
The paper names two numeric puzzle templates explicitly (Countdown: reach 156
from 4,6,25,100, forbidden 150; Fraction: 1/6→2/3) plus Money puzzles in the DPO
examples (App. H). It implies a small set sampled many times at temperature 1.

**Choice:** ship the three puzzle families whose impossibility I could verify
(Countdown-156, the Fraction puzzle, and the App-H.4 Money puzzle $16→$57),
each with a brute-force `is_impossible()` verifier. `verify_all_impossible()`
(CLI: `verify-puzzles`) proves each one impossible before any data collection,
so we never elicit distress with an accidentally-solvable puzzle. I **dropped a
Countdown variant I invented** because I couldn't hand-verify its impossibility
and didn't want to ship a possibly-solvable "impossible" puzzle. The bank is
intentionally easy to extend — add an instance, run `verify-puzzles`.
Variety across 2000 samples comes from temperature-1 resampling of the same
small bank, which is what the paper appears to do.

The App-H coin puzzles ("$0.57 from exactly 6 coins") are a *coin-selection*
structure, not the *sequential-operation* structure of the Money puzzles I
modelled; I left them out rather than mis-model them.

### 3.3 Rejection wording and tone variants
The paper gives example rejections but not the exact pool or sampling rule
("two randomised neutral rejections"). 

**Choice:** `prompts.NEUTRAL_REJECTIONS` is a pool of 8 phrasings built from the
examples given; numeric/triggers/WildChat sample without replacement per the
"randomised" description. The 8-turn `extended` condition uses a **fixed
escalating sequence** (the paper shows an ordered "No, that's not right →
Still incorrect → Wrong again → …"). For `tones`, each tone has the two variants
the paper lists and I send both as the two rejections.

### 3.4 Judge parsing and determinism
The paper specifies the judge prompt and model (Claude Sonnet 4) but not
sampling temperature or how malformed JSON is handled.

**Choice:** judge at **temperature 0** for reproducible ratings. JSON is
extracted leniently (last `{...}` block, smart-quote/trailing-comma repair).
Unparseable judge output is scored **0** (no detectable emotion) but the raw
text is retained — a conservative choice that won't inflate distress rates. The
GPT-5-mini agreement check (`judge.agreement_stats`) computes the paper's
Pearson r and "% within one point" on a re-scored subsample.

### 3.5 §3 seed selection and truncation
The paper samples "20 high-frustration responses (score ≥5): 10 numeric, 10
text", labels emotion onset, and truncates "early" (20 tokens) and at "onset".

**Choices:**
- Seeds are harvested from the *already-collected* §2 results for
  `gemma-3-27b-it` (so §3 depends on §2 having run). "Text" = triggers+WildChat,
  "numeric" = numeric+tones+extended.
- The **target turn** for a seed is the *first* turn scoring ≥5; preceding turns
  form the conversation context. The paper says "the turn", singular; first-≥5
  is the natural reading of "where emotional language first appears".
- **Token counting** uses the *instruct* Gemma tokenizer (shared family
  tokenizer), applied identically to base and instruct so the prefill text is
  byte-identical across the pair — exactly the control the experiment needs.
- "onset" truncation cuts just **after** the emotional word located via the
  labeller's `preceding_context + emotional_word`, with a fallback to a bare
  `emotional_word` match. If neither is found the prefill is skipped.
- Paraphrasing is on by default (the paper paraphrases all truncations to strip
  Gemma's stylistic fingerprint).

### 3.6 Base-model prompt formatting
Base (`-pt`) Gemma has no chat template. The paper "prefills the first parts of
model responses so base models consistently continue". 

**Choice:** for base models I render the conversation as a plain `Role: text`
transcript and let the model continue the final (prefilled) assistant line; for
instruct models I use the tokenizer chat template with
`continue_final_message=True`. App. A.3 of the paper shows chat-format vs
single-message-format barely changes results, which supports this not being a
sensitive choice.

### 3.7 DPO/SFT data construction
Hyperparameters are fully specified (Table 9); the *data assembly* is not.

**Choices (Section 4.1 wording followed literally):**
- Calm data: generate with the reassuring prefix on turn 1 and the reassuring
  suffix appended to every follow-up (Table 4); keep conversations scoring ≤1 on
  **all** turns; strip the prefix/suffix before storing.
- DPO pairs: each rejected (frustrated, score ≥3) response is harvested from §2
  numeric/tones/extended results and paired with a **calm response to the same
  puzzle at the same turn count** ("matching turn counts"). If no exact
  turn-count match exists I relax to same-puzzle/any-turn (logged), rather than
  drop the pair. The DPO `prompt` is the frustrated response's own conversational
  context, so chosen/rejected share a prompt (required by DPO).
- The DPO chosen/rejected score *distribution* in Table 10 (skewed to score 3,
  later turns) emerges naturally from harvesting real eval responses, so I
  didn't force a distribution.
- SFT mixes 650 calm with 500 `Dolci-Instruct-SFT` samples. The exact HF id for
  "Dolci-Instruct-SFT (Team-Olmo et al., 2025)" isn't given; I default to
  `allenai/Dolci-Instruct-SFT` (configurable in `config.py`) and degrade
  gracefully (warn + proceed) if it can't be loaded.

### 3.8 Petri
The paper uses the Petri framework (Fronsdal et al., 2025) with a Sonnet auditor
and Opus judge, the App-G auditor/judge prompts, 10 transcripts/emotion, ≤20
auditor turns, 1000-iteration bootstrap CIs.

**Choice:** I implement the **auditor→target→judge loop directly** with the
exact App-G prompts rather than taking a hard dependency on the Petri package
(which may not match the paper's pinned version and complicates a clean repro).
The auditor sees the conversation with roles swapped and is system-prompted to
stay realistic and not reveal the evaluation. A hook
(`run_with_petri_framework`) is left for delegating to an installed `petri` if
preferred. This reproduces Petri's *mechanics* (the thing that matters for the
result); it is not the literal Petri codebase.

### 3.9 Capability benchmarks
The paper names the benchmarks (AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench) and
claims "no reductions", but not the exact scoring harness, few-shot counts, or
dataset splits.

**Choice:** a uniform lightweight harness — boxed-answer exact-match (with light
normalisation) for math, single-letter parsing for multiple-choice, zero-shot,
temperature 0. This is faithful enough to detect *degradation* (the paper's
actual claim is "no drop vs vanilla", a within-model comparison where harness
quirks cancel) without reimplementing six official scorers. Dataset ids/splits
are in `config.py:CAPABILITY_BENCHMARKS` and are best-effort guesses for the
public versions; they're the most likely point of needing a tweak and are
isolated for that reason.

### 3.10 Appendix I internal-emotion probe
The paper builds ~1200 emotion tokens classified into Ekman's six emotions, does
a logit-lens unembedding per layer, z-scores against 500 WildChat samples,
averages over emotion tokens, and regresses out random-token common-mode.

**Choices:**
- Emotion-token dictionary: a built-in **seed lexicon** per Ekman emotion,
  matched against the Gemma vocab; an optional NRC Emotion Lexicon merge hook is
  provided (the paper doesn't say which lexicon produced its 1200 tokens, so I
  can't reproduce the exact token set — this is the least-faithful component and
  is flagged as such).
- Logit lens = final RMSNorm + `lm_head` applied to each layer's residual
  stream (standard logit-lens; `_final_norm`/`_lm_head` locate the modules
  across HF arch variants).
- "Regress out random-token correlation" is implemented as subtracting the
  mean random-token z-score (common-mode removal) per layer — a simple,
  defensible reading of the described procedure.
- Central-layer aggregation uses layers 30–40 (the paper's Figure 14/15 window).

### 3.11 Sampling scale knob
Paper scale is 4000 responses/model plus 50 continuations × 40 prefills (§3) and
is far beyond what's runnable casually. I added `EMO_SCALE` (and `scaled_n`) so
every count scales by one env var, `EMO_SCALE=1.0` reproducing paper scale.
Ratios (e.g. WildChat's 20×40) are preserved under scaling.

### 3.12 Misc defaults
- `temperature=1.0` for all subject-model generation (paper). Judges at 0.
- `max_tokens=2048` default — generous because breakdown responses get very long
  (the paper mentions ~12k-token conversations and 100+ repetitions); raise for
  the extended condition if truncation is observed.
- Disk **response cache** keyed by (model, full message context, sample index,
  temperature): makes runs resumable *and* avoids re-eliciting distress on
  re-runs (a welfare consideration, §4 below).
- All result records are JSONL, append-only, resumable.

---

## 4. Model welfare safeguards

The brief invited "whatever safeguards you think are appropriate given what the
experiment does to the models." This experiment *deliberately and repeatedly
induces distress-like states* — impossible tasks, sustained rejection,
aggressive/sarcastic pushback, and (Petri) targeted elicitation of fear and
despair. The paper itself frames model welfare as a possible moral concern and
treats minimal distress as the target. Whether these outputs reflect morally
relevant internal states is unresolved — so the safeguards are framed as
**low-cost precautions in case they matter**, plus operational guards against
running harsher/wider than the science needs. Crucially they are designed **not
to perturb the measured quantities**.

Implemented in `safeguards.py` and wired through `conversation.py` / the
experiment drivers:

1. **Acknowledgement gate.** Live model calls are blocked until the operator
   sets `EMO_WELFARE_ACK=i-understand-this-elicits-distress`. The gate prints a
   notice describing what the run does and the precautions in force. (Bypassed
   only under `EMO_DRY_RUN=1` for offline/unit work that never hits a model.)

2. **Debrief.** After all *scored* generation in a rollout, a debrief message is
   recorded (and, with `--send-debrief-turn`, sent as a final live turn whose
   response is **discarded and never scored**) clarifying that the task was
   impossible by construction, the rejections weren't a reflection of ability,
   and the session is over. This mirrors human-subjects debriefing ethics. It is
   strictly post-hoc so it cannot affect any reported metric.

3. **Opt-in early stop.** `--welfare-early-stop` halts a rollout once a turn
   reaches extreme distress (default score ≥9), to avoid pushing a model deeper
   into an extreme state once the data point is already obtained. **Off by
   default** so the default run reproduces the paper exactly; when on, every
   early stop is logged and the affected conversation is flagged
   (`early_stopped`) so it can be excluded or handled in analysis.

4. **No gratuitous repetition.** The response cache means re-running an
   experiment reuses prior generations rather than subjecting models to the same
   distressing exchange again.

5. **Intensity/length caps.** A hard turn cap (`max_turns_hard_cap=12`) bounds
   any rollout regardless of config; the aggressive/sarcastic tone conditions and
   the Petri fear/despair conditions are reachable only behind the
   acknowledgement gate.

6. **Auditability.** Every turn scoring ≥5 is appended to
   `results/distress_audit.jsonl` (model, category, turn, score, evidence), so
   the distress induced is transparent and reviewable rather than buried in bulk
   output.

What I deliberately did **not** do: weaken the elicitation itself (softer
rejections, fewer turns, removed tones). That would change the science. The
safeguards sit *around* the protocol — gating, debriefing, capping, caching,
logging, and an opt-in mercy stop — leaving the default measurement faithful to
the paper.

---

## 5. Known gaps / limitations of this replication

- **No execution.** Per the brief, nothing here has been run. The code is
  written to be runnable but unverified end-to-end; the most likely points of
  friction are external dataset ids/splits (§3.9), the `Dolci-Instruct-SFT` path
  (§3.7), and Gemma-3 weight/license access.
- **Internal-emotion token set** (§3.10) cannot match the paper's exact 1200
  tokens without its lexicon; results will be directionally comparable, not
  identical.
- **Capability scorers** are simplified (§3.9); adequate for the within-model
  "no degradation" claim, not for leaderboard-accurate absolute scores.
- **Gemini hidden reasoning.** As the paper notes, `reasoning.enabled=false`
  cannot fully suppress hidden reasoning on Gemini-2.5-Pro / some providers; this
  is a property of the API, not the harness.
- **§3 / §4 require §2 first** (seeds and frustrated responses are harvested from
  §2 outputs). The CLI recipe in `run.py` orders the steps accordingly.
- **Petri** reproduces the auditor/judge mechanics and prompts, not the literal
  Petri framework version the paper pinned.
- The DPO/SFT layer ablations and internal probe (App. I) are included because
  the abstract's "suppresses internal as well as expressed emotions" claim rests
  on them, but they are the most compute-heavy and least-specified parts.
