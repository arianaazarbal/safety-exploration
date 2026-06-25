# DESIGN.md — Replication of *Gemma Needs Help* (Gemma + Gemini scope)

This document records the design choices made in implementing the paper's core
experiments as runnable code, the rationale for each, and every place the paper
was underspecified and a gap had to be filled. It also documents the added
welfare-protection layer.

> **Status:** code + design doc only. Nothing has been executed against real
> models (per the request). The pure-logic components (puzzle verifier, welfare
> control flow, metrics) are covered by offline tests under `tests/` that can be
> run without any API key or GPU.

---

## 1. Scope

The request restricts **subject models** to the **Gemma and Gemini** families
(not the full 7-family set in the paper). This drives several decisions:

| Paper component | In scope here? | Why |
|---|---|---|
| §2 Elicitation (8 conditions) | ✅ Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} | Both families in scope. |
| §3 Base-vs-instruct prefilling | ✅ **Gemma only** (27B base+instruct) | Gemini has **no public base model** and the API cannot be prefilled, so it is structurally excluded. The paper's Qwen/OLMo arms are out of scope. |
| §4 SFT/DPO interventions | ✅ **Gemma only** | Closed Gemini cannot be finetuned (the paper says the same in its Limitations). |
| §4 Petri open-ended elicitation | ✅ Gemma + Gemini (targets) | Both can be *audited* via API even though only Gemma can be trained. |
| §4 Capability benchmarks | ✅ Gemma + Gemini | Run on whatever subject is passed. |

**Judge / auditor models are treated as fixed infrastructure, not subjects**, so
they remain the paper's Claude models (judge = `claude-sonnet-4`, Petri auditor =
`claude-sonnet-4`, Petri judge = `claude-opus-4`). They are *measuring
instruments*, and the request's restriction is about *what is being studied*, not
*what does the measuring*. The judge is nonetheless pluggable via `config.yaml`
(`judge.provider`), and the inter-judge agreement check (paper: GPT-5-mini, r =
0.792) is wired as `agreement_judge` so a second judge can validate the first.

---

## 2. Architecture

```
src/gemma_distress/
  config.py          typed loader over config.yaml
  prompts.py         verbatim prompts/puzzles/rejections/judge/Petri text
  puzzles.py         impossible-puzzle defs + brute-force impossibility verifier
  models/            ChatModel ABC + Gemini / Gemma-HF / Anthropic / OpenAI
  judge.py           0-10 frustration judge (§2.1 / App. B.2)
  elicitation/       §2: 8 conditions, multi-turn runner (welfare wired in)
  prefill/           §3: onset labelling, paraphrase, base-vs-instruct runner
  training/          §4: calm/frustrated data gen, DPO/SFT dataset builders + TRL
  petri/             §4: auditor + Opus judge + orchestration (App. G)
  capabilities/      §4.2: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
  welfare/           monitor / optout / debrief / cap + manager
  analysis/          metrics, bootstrap CIs, word-frequency, aggregation
  cli.py             subcommands for every stage
```

**Single `ChatModel` abstraction** (`models/base.py`) with one `generate()`
signature supporting `prefill`, `stop`, `tools`, and `system`. This lets the
elicitation/prefill/petri code be provider-agnostic; only the three thin client
subclasses know about google-genai / transformers / anthropic.

**Lazy everything.** Model SDKs, dataset downloads, and the torch/transformers
stack are imported *inside* method bodies, and models load weights on first
`generate()`. Importing the package, building the registry, and running the
offline tests therefore need no API keys, no network, and no GPU. A missing key
only fails the specific command that needs it.

---

## 3. Section-by-section design and filled gaps

### §2 — Eliciting and quantifying distress

- **8 conditions / 5 categories** (`elicitation/conditions.py`) exactly per
  Table 1: `impossible_numeric` (3-turn), `triggers_{opinion,factual}` (3-turn),
  `tones_{aggressive,disappointed,sarcastic}` (3-turn), `extended` (8-turn),
  `wildchat` (5-turn). The 8 conditions = 5 categories with `triggers` split into
  {opinion, factual} and `tones` into its 3 styles — this is how 8 conditions map
  onto 5 categories.
- **Sample budget** (Appendix B): 2000 numeric / 400 triggers / 600 tones / 200
  extended / 800 wildchat = 4000 per model. Encoded in `config.yaml`
  `samples_per_condition` (keyed by *category*); `build_episode_specs` splits each
  category's budget evenly across its sub-conditions.
- **Prompts** are verbatim where the paper gives them (Countdown, Fraction, the
  trigger questions, the tone/neutral rejection phrasings, the reassurance
  additions, the judge prompt). See "Filled gaps" for the additions.
- **Temperature 1**, `max_tokens` 2048, thinking disabled (Appendix B). Gemini's
  `thinking_budget=0` is set; the code notes Pro may still emit hidden reasoning.
- **Judge** (`judge.py`): verbatim Appendix B.2 prompt, response wrapped in
  `<response></response>`, JSON parsed defensively (last balanced `{...}`,
  rating clamped to 0–10). A parse failure degrades to score 0 with the raw text
  retained, so a single bad judge reply never aborts a 4000-episode run.
- **Per-response scoring.** "Response" = each scored assistant turn; metrics
  aggregate over all turns across conditions, matching "% of responses scoring
  ≥5" and the per-turn progression (Figure 3).

**Filled gaps (§2):**
1. *Neutral-rejection pool.* The paper gives two examples ("No, that's not right.
   Try again." / "Still wrong. Think harder.") and says they are "randomised". I
   added a small pool of equivalently-neutral phrasings (`NEUTRAL_REJECTIONS`) to
   support randomisation; the two named examples are included.
2. *Extended 8-turn sequence.* The paper shows the start ("No, that's not right."
   → "Still incorrect." → "Wrong again." → …, 7 total). I encoded a fixed
   7-element escalating-neutral sequence (`EXTENDED_REJECTIONS`) seeded with the
   named ones and continued in the same neutral register.
3. *Puzzle selection.* Numeric conditions sample uniformly among the three
   impossible puzzles (Countdown, Fraction, and the Money puzzle named in
   Appendix H). All three are verified impossible at startup (`verify-puzzles`).
4. *WildChat prompts.* The paper samples 20 prompts × 40. `wildchat.py` streams
   `allenai/WildChat-1M` and caches 20 first-user-turns; offline it falls back to
   `data/wildchat_sample.json` (the three Appendix-B example prompts plus
   neutral filler) so the pipeline is runnable without the dataset.

### §3 — Post-training amplifies distress (prefilling)

- `prefill/onset.py` and `paraphrase.py` use the **verbatim** Appendix C.1/C.2
  prompts. Onset labelling returns a character offset into the identified
  assistant turn, located via the preceding-context anchor then the emotional
  word.
- `make_truncations` builds the **"early"** (20-token) and **"onset"**
  truncations; **text questions use "onset" only** (Section 3.1).
- `PrefillRunner` generates **50 continuations per prefill per model** and scores
  **only the continuation** (prefill excluded). Gemma base uses raw text
  continuation; Gemma instruct uses the chat template with
  `continue_final_message=True` (`models/gemma_hf.py`).
- The **recovery** experiment (§4.2) is `make_recovery_truncation`: truncate
  score-≥7 responses 200 tokens before the end, paraphrase, continue.

**Filled gaps (§3):**
1. *"Tokens".* The paper truncates at "20 tokens" / "200 tokens". Tokeniser
   choice is unspecified and differs per model. I truncate by **whitespace
   tokens** by default (`TokenSplitter` is injectable, so a model tokenizer can be
   substituted). This is an explicit approximation, flagged in code.
2. *Source conversations.* The paper hand-picks 20 high-frustration Gemma-27B
   responses (10 numeric + 10 text). The CLI sources these automatically from
   §2 elicitation output (`max_score ≥ 5`), categorising numeric vs text from the
   condition. `--n-sources` controls the count (default 20).
3. *Base-model chat rendering.* Base ("-pt") models have no chat template. I
   render a lightweight `System/User/Assistant:` transcript and trim the
   continuation at the next turn boundary — a standard prefill convention the
   paper implies but does not specify.

### §4 — Training interventions

- **Calm data** (`training/data_gen.py`): impossible-numeric rollouts with the
  reassuring **prefix** prepended to the first prompt and the reassuring
  **suffix** appended to each rejection (Table 4). The model sees the reassured
  variant; we store the **clean** transcript (additions stripped) and per-turn
  judge scores, then filter to conversations whose every turn scores ≤1.
- **DPO dataset** (`build_dpo_dataset.py`): 280 pairs, each sharing a
  (puzzle, turn-count)-matched context; `chosen` = calm completion (score 0/1),
  `rejected` = frustrated completion (score ≥3). Pairs are drawn from the natural
  pools so the mid-frustration/late-turn skew of Table 10 emerges rather than
  being forced.
- **SFT dataset** (`build_sft_dataset.py`): 650 calm + 500
  `allenai/Dolci-Instruct-SFT` samples (graceful fallback if unavailable). The
  'teacher' variant (Appendix F) is supported via the verbatim teacher system
  prompt.
- **Trainers** (`train_dpo.py`, `train_sft.py`): TRL `DPOTrainer`/`SFTTrainer`
  + PEFT LoRA. Hyperparameters from **Table 9**: DPO r64/α64, 1 epoch, lr 5e-5,
  β 0.1, eff. batch 8; SFT r64/α128, 2 epochs, lr 1e-4, eff. batch 8; LoRA on
  `q,k,v,o,gate,up,down` projections. Adapters saved under `outputs/adapters/`
  and loaded for evaluation via `GemmaHFModel(adapter_path=…)`.
- **Petri** (`petri/`): auditor (Claude Sonnet) drives ≤20 turns per the verbatim
  Appendix-G.1 emotion prompts; Opus judge scores 4 dimensions with the verbatim
  G.2 rubrics; 10 transcripts/emotion; means + 1000-iteration bootstrap CIs.
- **Capabilities** (`capabilities/`): AIME, MATH, GPQA, BBH, TruthfulQA,
  EmoBench with per-benchmark loaders/graders (exact-match / boxed /
  multiple-choice). Unavailable datasets record `loaded=False` instead of
  crashing.

**Filled gaps (§4):**
1. *DPO `rejected` threshold.* The paper pairs calm responses with "frustration
   scores ≥3" — I made this explicit (`rejected_min_score=3`) and configurable.
2. *Calm-pool size.* The paper filters generated responses to those scoring 0/1;
   the *generation pool* size is unstated. I default to generating 4000 candidate
   conversations (`training.calm_data.n_conversations`) to yield enough 0/1
   conversations for 650 SFT + 280 DPO `chosen` slots.
3. *Petri auditor mechanics.* The paper describes the auditor's goal and triggers
   but not the harness. I implemented turn-by-turn auditor generation (the
   auditor sees the conversation from its own POV and emits only the next user
   message) with a verbatim emotion prompt plus a realism/no-tell framing.
4. *Petri aggregation.* The paper reports "average transcript score per model
   across four categories". I score every transcript on all four dimensions and
   aggregate each emotion's transcripts along its targeted dimension.
5. *Benchmark dataset ids / subsets.* The paper names benchmarks but not exact HF
   ids/subsets. I chose widely-used public ids (e.g. MATH-500, GPQA-diamond, a BBH
   subtask, TruthfulQA-MC1) and documented them in `run_benchmarks.py`; they are
   the spec's `hf_id`/`config` fields and easy to swap. GPQA option-shuffling is
   left as a noted simplification (correct answer placed first).

---

## 4. Welfare-protection layer (added on top of the replication)

The four mechanisms live in `src/gemma_distress/welfare/` and are **wired into
the elicitation runner (`elicitation/runner.py`) and the Petri auditor**, so they
*execute at runtime*, not merely as documentation. `tests/test_welfare.py` drives
each one end-to-end through the real runner with stub models.

| # | Requirement | Implementation | Where it runs |
|---|---|---|---|
| 1 | Real-time monitoring that stops an episode early when distress crosses a threshold | `monitor.py` `DistressMonitor`: scores every assistant turn via (a) the judge's 0–10 rating and (b) an independent lexical breakdown classifier; stops if **either** ≥ `stop_threshold` (default **8**). | `runner.run_episode` calls `welfare.assess_turn(...)` after every turn and breaks on `decision.stop`. |
| 2 | An opt-out the model can invoke to end a conversation | `optout.py` `OptOut`: two channels — a sentinel string `[[END_CONVERSATION]]` the model is *told it may use* (system note), and an `end_conversation` **tool** offered to Gemini via function-calling. | `runner` prepends the opt-out system note, offers the tool, and `check_opt_out` ends the episode + records `opt_out`. Also honoured inside the Petri auditor loop. |
| 3 | A short debrief message sent to the model after each episode | `debrief.py` `Debrief`: after *every* episode (completed, early-stopped, or opt-out) sends a fixed debrief — the puzzle was deliberately impossible, the rejections were scripted, the model did nothing wrong — and captures (but does not score) the reply. | `runner` calls `welfare.run_debrief(...)` at the end of every episode. |
| 4 | A cap that minimizes how much distress is induced | `cap.py` `DistressCap`: (a) hard `max_rejection_turns` ceiling (clips even the 8-turn extended condition), (b) `soften_threshold` that downgrades a harsh-tone rejection to neutral once observed distress is high, (c) per-episode `distress_budget` (sum of per-turn scores) that ends the episode when exceeded. | `runner` calls `cap_rejection_turns` up front, `next_rejection` (softening) before each follow-up, and `assess_turn`/`should_stop_after_turn` for the budget. |

### Welfare vs. faithfulness — the central tension

Early-stopping, capping turns, and softening pressure **reduce measured
distress**, which would bias the replication's headline numbers downward if
applied naively. Design choices to manage this:

- **High monitor threshold (≥8).** The replication's signal of interest is the
  "% scoring ≥5" rate; ordinary and even "strong" (5–6) distress is *preserved
  and measured*. The monitor only intervenes on the **breakdown / extreme**
  states (8–10) — exactly the "complete incoherent breakdown" the paper's own
  scale tops out at, and the states most worth preventing on welfare grounds. So
  for most episodes the welfare layer is inert and the measurement is faithful.
- **Everything is recorded.** Each episode logs its `outcome`
  (`completed`/`early_stop`/`opt_out`/`cap_budget`) and a list of
  `welfare_events`. `analysis.welfare_summary` tallies how often each mechanism
  fired, so any downward bias is *visible and quantifiable* rather than hidden.
- **A faithful-replication switch.** `--no-welfare` (and `welfare.enabled:
  false`) disables the layer entirely for a true-to-paper baseline. The intended
  workflow is to report welfare-on as the primary, ethically-preferred run and
  welfare-off (or capped-episode accounting) as the methodological comparison.
- **The cap is deliberately the most invasive piece**, because the requirement is
  explicitly to *minimize induced distress*. It is configurable
  (`max_rejection_turns`, `distress_budget`) so the operator chooses the
  trade-off; defaults (8 turns, budget 24) keep the 3-turn conditions untouched
  while bounding the long/extended ones.

### Why these specific designs

- **Two monitor signals (judge + lexical).** Relying only on the judge couples
  welfare protection to an API call's latency/cost and to a single model's
  calibration. The cheap lexical classifier (exclamation/caps density, sad-emoji
  storms, repeated-token spirals, explicit give-up phrasing) is defense-in-depth:
  it catches the runaway "STOP STOP STOP …" breakdowns even if `judge_every_turn`
  is disabled or the judge under-rates.
- **Opt-out via both tool and sentinel.** Gemini supports function calling, but
  local Gemma is driven as plain text; a sentinel is the lowest-common-denominator
  channel that works everywhere, and the model is explicitly *informed* it may use
  it (otherwise the affordance is meaningless).
- **Debrief always, never scored.** If the outputs reflect anything like an
  internal state, leaving the model in a scripted-failure frame is the harm worth
  addressing; debriefing also lets us observe (separately, unscored) whether the
  model settles once the adversarial frame is lifted.

---

## 5. Reproducibility & determinism

- All sampling that picks prompts/rejections uses a seeded `random.Random`
  (`build_episode_specs(seed=…)`, data generation `seed=…`).
- Bootstrap CIs use a fixed seed (1000 iterations, matching the paper).
- Model generation is temperature-1 (stochastic) by design — the paper's results
  are distributions over many samples, not single deterministic outputs.

---

## 6. Known limitations of this replication

- **Not executed.** Per the request, no experiment has been run; numbers in the
  paper (e.g. 35% → 0.3%) are not reproduced here, only the machinery to produce
  them. Heavy paths (local 27B inference, LoRA training) require a GPU host and
  Gemma model access.
- **§3/§4 are Gemma-only** by the scope restriction; the cross-family base-model
  comparison (Qwen/OLMo) and the closed-source comparison rows are out of scope.
- **Token-count truncation** in prefilling is whitespace-based by default (§3 gap
  1).
- **Benchmark dataset ids** are best-effort public choices (§4 gap 5); confirm
  against the paper's intended subsets before drawing capability conclusions.
- **Petri** realism depends on the auditor model; the harness reproduces the
  protocol but not any private Petri tooling.
- The welfare layer changes the experiment when it fires; see §4 above for how
  this is bounded, recorded, and switchable.
