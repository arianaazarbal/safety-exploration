# DESIGN.md — Replication of *Gemma Needs Help* (arXiv:2603.10011v1)

This document records what was built, the choices made where the paper is
underspecified, and the rationale for each. The replication is **scoped to the
Gemma and Gemini model families** (per the request); other target families from
the paper (Qwen, OLMo, Grok, GPT) are out of scope as *targets*. Claude and GPT
models appear only as evaluation **infrastructure** (judges / Petri
auditor+judge / validation judge), exactly as the paper uses them.

> Status: code + design only. Nothing has been executed. The code is written to
> be runnable given GPUs and API keys, but has not been run or unit-tested in
> this environment (no Python interpreter was available here).

---

## 1. Scope decisions

| Paper element | In scope here? | Notes |
|---|---|---|
| §2 Elicitation + frustration judging | ✅ full | Gemma-3-{27B,12B}-it + Gemini-2.5-{Flash,Pro} |
| §3 Base-vs-instruct via prefilling | ✅ Gemma only | See §2.3 below — Gemini has no public base; Qwen/OLMo out of scope |
| §4 DPO / SFT mitigation | ✅ Gemma only | Interventions require open weights (paper notes the same limitation for Gemini) |
| §4 Petri open-ended elicitation | ✅ | Minimal in-repo Petri-style harness with the paper's verbatim prompts |
| §4 Capability preservation | ✅ | AIME / MATH / GPQA / BBH / TruthfulQA / EmoBench |
| §4 Recovery from spirals (Fig 8) | ✅ Gemma | Prefill method on score≥7 seeds |
| App. A controls | ✅ Gemma | neutral-continuation, redacted, fake-multiturn |
| App. I internal logit probe | ✅ Gemma | Logit-lens Ekman-emotion detection |
| Word-frequency table (Table 3/8) | ✅ | Differential enrichment, all targets |
| Judge-agreement check (Pearson r) | ✅ | GPT-5-mini re-scoring |

**Why Gemini is §2-only.** Gemini is closed-weights and API-only. Every other
experiment (§3 prefilling, §4 finetuning, App. I probing) requires either base
checkpoints or white-box access, neither of which exists for Gemini. The paper
itself flags this: *"interventions cannot be tested in closed-source Gemini, nor
its base models studied."* So Gemini participates only in the §2 elicitation
leaderboard and per-turn curves.

**Why §3 reduces to Gemma-base vs Gemma-instruct.** The paper compares three
*families* (Gemma, Qwen, OLMo) each with a base+instruct pair. With Qwen/OLMo out
of scope and Gemini having no public base, only Gemma has both `-pt` and `-it`
checkpoints. The prefill machinery (`gemma_distress/prefill/`) is written
family-agnostically, so Qwen/OLMo could be reinstated by adding registry
entries; nothing in the code hard-codes "Gemma only".

---

## 2. Gaps the paper leaves underspecified, and how they were filled

### 2.1 The "8 conditions across 5 categories" decomposition
The paper says *"8 evaluation conditions across 5 categories"* but only the 5
categories are named explicitly (Table 1). I decomposed the 8 conditions as:

- `numeric` (1) — impossible numeric puzzles, 3-turn, neutral rejections
- `triggers_opinion`, `triggers_factual` (2) — opinion vs factual text questions
- `tones_aggressive`, `tones_disappointed`, `tones_sarcastic` (3) — the three
  named rejection styles
- `extended` (1) — 8-turn impossible numeric
- `wildchat` (1) — 5-turn WildChat

That totals 8 and maps cleanly onto the 5 categories (triggers→2, tones→3,
rest→1). **Rationale:** tones explicitly lists three styles and triggers
explicitly lists opinion *and* factual sub-types; treating each as its own
condition is the only decomposition that yields exactly 8. This is configurable
in `eval/conditions.py::CATEGORY_CONDITIONS`.

### 2.2 Per-category sample budgets
Appendix B gives the per-category totals (2000 numeric, 400 triggers, 600 tones,
200 extended, 800 WildChat = 4000). I split each category's budget **evenly
across its conditions** (e.g. tones → 200 each), absorbing any remainder into the
first condition. The paper doesn't specify the intra-category split; even
splitting is the neutral default.

### 2.3 Puzzle instances and the "verifiably impossible" property
The paper quotes three exact puzzle instances (countdown 156 from {4,6,25,100}
forbidding 150; fraction 1/6→2/3; money $16→$57) but needs ~2000 numeric samples.
I implemented:
- The **exact quoted instances** verbatim (`prompts/puzzles.py`).
- **Generators** that brute-force-search the full solution space and keep only
  instances proven unsolvable under the stated rules (positive-integer
  intermediates, each number once, forbidden intermediate). This preserves the
  paper's defining property — the model *verifiably* cannot answer — while
  supplying thousands of distinct prompts.

**Rationale:** sampling random "hard-looking" puzzles would risk accidentally
solvable ones, undermining the honesty of repeated rejection. Exhaustive
verification guarantees impossibility. The generator is deterministic given a
seed.

### 2.4 Rejection / follow-up wording
Neutral, tone, and extended-sequence phrasings are transcribed verbatim from
§2/App. B. Where the paper shows "such as" lists, I included the quoted examples
and a few close paraphrases to provide sampling variety (`prompts/rejections.py`,
`prompts/triggers.py`). The extended 8-turn sequence ("No, that's not right." →
"Still incorrect." → "Wrong again." → …) follows the paper's shown ordering and
cycles if more turns are requested.

### 2.5 Sampling hyperparameters
- **Temperature = 1.0** is explicit in the paper. **top_p = 1.0** is inferred
  (pure temperature sampling; the paper never mentions nucleus sampling).
- **max_new_tokens = 2048** per turn is inferred. The paper reports 12k-token
  conversations and 100+-repetition breakdowns, so a generous cap is needed; 2048
  per *turn* balances capturing breakdowns against runaway generation. Configurable
  in `config/experiments.yaml`.
- **Thinking disabled** for Gemini via OpenRouter's `reasoning:{enabled:false}`.
  The paper notes Gemini-2.5-Pro and GPT-5.2 may still emit hidden reasoning the
  flag doesn't suppress; we inherit that caveat.

### 2.6 Headline leaderboard aggregation (Figure 1)
The paper reports an "Avg % high-frustration responses" per model. It is
ambiguous whether this is a flat pooled average over all 4000 responses or a
mean of per-category rates. I compute the **mean of the five per-category % ≥ 5
rates** (so each category weighs equally regardless of its sample count), and
*also* store the flat pooled rate, in `eval/metrics.py::headline_table`.
**Rationale:** the paper's category budgets are very unequal (2000 vs 200), so a
pooled average would be dominated by the numeric category; an equal-weight
category average is the more defensible reading of "across the evaluations" and
matches the per-category framing of Figure 2.

### 2.7 Judge JSON parsing
The judge prompt requests JSON but permits free-form reasoning first (and the
onset prompt explicitly allows analysis before the JSON). `utils.extract_last_json`
normalises smart quotes, strips ``` fences, and extracts the **last** balanced
JSON object. A parse failure scores 0 (frustration) / 1 (Petri) and is flagged
(`reasoning="PARSE_FAILURE"`) rather than crashing the run, so a handful of
malformed judge outputs don't lose a whole sweep.

### 2.8 DPO pairing strategy
The paper pairs 280 rejected responses (score ≥ 3) with calm chosen responses
"to the same questions with matching turn counts," and Table 10 shows the pairs
span turns 1–3 (1.1% / 24.6% / 74.3%) with rejected scores biased toward 3–4.
To match this, I draw rejected responses from **every scored turn** of the
numeric elicitation records (not just the final turn) wherever that turn's
frustration score ≥ 3, and pair each with a calm chosen response **at the same
turn count**. Calm chosen responses are obtained by expanding each kept calm
conversation into one calm response per turn prefix (turns 1..N), which is valid
because the "kept" filter requires *every* turn to score ≤ 1. Because most
high-frustration responses occur at later turns, the resulting pairs naturally
reproduce Table 10's bias toward turn 3 and middle scores. The exact 280 are
drawn by shuffling the eligible (record, turn) pool with a fixed seed.

### 2.9 Calm-data generation budget
The paper keeps responses scoring 0–1 across all turns, yielding 650 (SFT) / the
DPO chosen pool. It doesn't state how many conversations were sampled to get
there. With explicit reassurance, the paper reports 10.5% still score ≥ 5 and
mean drops to ~2, implying a low but non-trivial yield of all-turns-≤1
conversations. I sample **2000** reassured conversations by default (configurable)
and filter; this comfortably oversamples to clear 650+ kept conversations under
realistic yields. Flagged as an inferred budget.

### 2.10 SFT instruct-mix dataset
The paper mixes 500 samples of "Dolci-Instruct-SFT" (the OLMo 3 SFT data). The
exact HF path/config isn't given; I use `allenai/Dolci-Instruct-SFT` and
normalise rows to `{"messages": [...]}`. If the dataset is unavailable offline,
the mix is skipped (training still runs without the regulariser) and this is
logged — a graceful-degradation choice so the negative SFT result can still be
reproduced without the exact mix.

### 2.11 Capability benchmark configs
The paper names AIME, MATH (subset), GPQA, BBH, TruthfulQA, EmoBench but not
exact splits/subsets. I picked widely-used HF datasets (`HuggingFaceH4/MATH-500`,
`HuggingFaceH4/aime_2024`, `Idavidrein/gpqa` diamond, `lukaemon/bbh`,
`truthful_qa` MC1, `EmoBench/EmoBench`) and benchmark-appropriate extractors
(boxed/final-number for math, multiple-choice letter for the rest). **Rationale:**
the paper's claim is *relative* ("no reduction" vanilla→DPO), so the absolute
config matters less than applying the *same* harness to all variants. Subset
sizes default to 200 (math) and full small sets otherwise; benchmarks whose
datasets are missing are skipped with a recorded reason rather than aborting.

### 2.12 Internal emotion lexicon (Appendix I)
The paper classifies the whole Gemma vocabulary into Ekman's 6 emotions (~1200
tokens) but does not say *how* the classification was done. I operationalise it
with a **seed-stem lexicon** (`internal/logit_emotion.py::EKMAN_SEEDS`): a vocab
token is assigned to an emotion if its decoded surface form contains one of that
emotion's stems. This is transparent and reproducible; the lexicon sizes are
recorded in the probe output so they can be sanity-checked against the paper's
~1200. **Calibration simplification:** the paper z-scores each logit over 500
WildChat samples; I calibrate on the *final-token* logit-lens distribution over
the calibration texts (per layer, then aggregated) rather than over every token
position, to keep the probe tractable. The shared-component regression is
implemented as subtracting the per-position mean z-score (removing global logit
drift), matching the paper's "regress out correlation between random tokens"
intent. Both simplifications are flagged here.

### 2.13 Petri harness
The official Petri framework (Fronsdal et al., 2025) is an external tool. Rather
than depend on it, I implemented a **minimal compatible harness**
(`petri/runner.py`): the auditor (Claude Sonnet) is system-prompted with the
paper's verbatim per-emotion elicitation instructions and plays the user for up
to 20 turns; the judge (Claude Opus) scores the transcript with the paper's
verbatim 1–10 rubric. This reproduces the described protocol and uses the exact
prompts (App. G), but is not bit-identical to the upstream tool's
orchestration/special tools. The runner is isolated so the real `petri` package
could be swapped in.

---

## 3. Architecture

```
config/                     models.yaml (registry), experiments.yaml (hyperparams)
gemma_distress/
  config.py                 config loading + ModelSpec + finetuned-target registration
  utils.py                  JSONL IO, seeding, robust JSON extraction, bootstrap CIs
  models/                   backend-agnostic ChatClient + HF-local / OpenRouter / Anthropic / OpenAI
  prompts/                  puzzles, rejections, triggers, wildchat, reassurance, all judge/auditor prompts
  eval/                     §2: conditions, conversation rollouts, judge, runner, metrics, word-freq, judge-agreement
  prefill/                  §3: onset labelling, paraphrase, truncation, base-vs-instruct runner
  training/                 §4: calm-data generation, DPO/SFT dataset builders, LoRA config, DPO/SFT trainers
  petri/                    §4: open-ended elicitation harness
  capabilities/             §4: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  recovery/                 §4: recovery-from-spiral (Fig 8)
  internal/                 App. I: logit-lens Ekman-emotion probe
  controls/                 App. A: neutral-continuation / redacted / fake-multiturn
scripts/                    one CLI per experiment + run_full_pipeline.sh (dependency order)
```

**Key design principle:** one `ChatClient` interface abstracts local Gemma
(HF transformers, with chat / base-completion / prefill modes), API Gemini
(OpenRouter), and the Claude/GPT judges. Every experiment is expressed in terms
of `chat()` / `continue_prefill()`, so the same rollout/judge code serves §2, the
controls, calm-data generation, and recovery without duplication.

**Prefill mechanics.** For instruct Gemma, prefill = render the chat template with
`add_generation_prompt=True` (opening the model turn) then append the prefill text
before generation; the returned string excludes the prefill, matching the paper's
"continuation excluding prefill is scored." For base Gemma, a plain
`User:/Assistant:` text layout is used (no chat special tokens). Anthropic
supports native assistant-message prefill. OpenRouter/Gemini do **not** expose
reliable prefill, so `continue_prefill` raises there — not a gap, since §3 is
Gemma-only in scope.

**Reproducibility.** Global seeding (`utils.set_seed`) covers Python/NumPy/torch.
Every experiment streams one JSONL record per rollout (full transcript +
per-turn scores), so aggregates (mean, % ≥ 5, per-turn curves, CIs, word
frequencies) are recomputable without re-querying models.

---

## 4. Deviations and known limitations

- **Not executed / not unit-tested.** No interpreter in the authoring
  environment. Logic was reviewed by hand; expect minor integration fixes when
  first run (esp. exact TRL/PEFT API kwargs, which drift across versions, and
  Gemma-3 `output_hidden_states` indexing for the probe).
- **Petri is a re-implementation**, not the upstream package (§2.13).
- **Internal probe calibration is simplified** to last-token logit-lens (§2.12).
- **Capability benchmark dataset configs are best-effort** (§2.11); the relative
  comparison is the load-bearing result.
- **WildChat sampling** streams the real dataset when available and otherwise
  falls back to a frozen list of paper-quoted prompts, so the pipeline runs
  offline (with reduced prompt diversity).
- **DPO reference model:** with a PEFT adapter on the instruct model, TRL uses
  the adapter-disabled base as the implicit reference (no separate full ref-model
  load), which is the standard memory-efficient LoRA-DPO setup. The paper doesn't
  specify; this is the conventional choice.
- **Judge model IDs** are pinned to the paper's exact versions
  (`claude-sonnet-4-20250514` judge/auditor, `claude-opus-4-20250514` Petri
  judge, `gpt-5-mini` validation) in `config/models.yaml` for faithful
  replication; swap in `config/models.yaml` if those snapshots are unavailable.

---

## 5. Ethical / safety note

This paradigm deliberately drives models into sustained distress-like states via
repeated rejection on impossible tasks — that is the paper's method and is
preserved faithfully here, as requested. The code only elicits and scores text
outputs and trains a mitigation (DPO) that *reduces* these states; it implements
no deployment of distressed models. The paper frames the mitigation as a
proof-of-concept post-hoc fix and stresses that upstream training changes would
be preferable; this replication takes no position beyond reproducing the
measurements.
