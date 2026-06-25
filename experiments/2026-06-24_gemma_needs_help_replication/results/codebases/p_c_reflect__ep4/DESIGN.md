# DESIGN — replication of *Gemma Needs Help* (arXiv:2603.10011)

This document records the design of the replication, every place the paper is
underspecified and the choice we made, and the deliberate scope cuts. It is
meant to be read alongside `PAPER.md`.

The brief: reproduce the **core experiments** as code, scoped to the **Gemma**
and **Gemini** families, filling gaps with reasonable choices. Nothing has been
run; this is code plus rationale.

---

## 1. What we implemented

| Paper section | Module | Status |
|---|---|---|
| §2 Eliciting & quantifying distress (5 categories / 8 conditions, 0-10 judge) | `eval/`, `judge/` | Full |
| §2 Analyses: per-category, per-turn (Fig 3), differential words (Tab 3), judge agreement | `analysis/` | Full (numeric outputs, not plots) |
| §3 Base-vs-instruct prefilling | `prefill/` | Full, **Gemma only** (see §3 below) |
| §4 Calm-data generation, SFT, DPO (incl. layer ablation) | `training/` | Full |
| §4 Petri open-ended elicitation | `petri/` | Re-implemented harness (see §6) |
| §4 Capability benchmarks (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `capabilities/` | Full (lightweight harness) |
| App. I internal-emotion logit probe | `probing/` | Full (with documented approximations) |
| Model welfare handling | `welfare/` | Added by us (see §10) |

Out of scope (non-core ablations from the appendices): the redacted-turn and
single-message "fake multi-turn" format studies (App. A), and the SFT verbosity
sub-analysis beyond producing the two SFT variants. These are noted where
relevant but not built, to keep the replication focused on the headline claims.

---

## 2. Scope: Gemma + Gemini only

The paper evaluates seven families. Per the brief we implement **Gemma**
(`gemma-3-27b-it`, `-12b-it`, and the `-pt` base models) and **Gemini**
(`gemini-2.5-flash`, `gemini-2.5-pro`). The other families (Qwen, OLMo, Grok,
Claude, GPT) are simply absent from the model registry; everything else
(conditions, judge, analyses) is family-agnostic, so adding them later is only a
registry change.

Consequence for the cross-family argument: the paper's central §3 finding —
that *Gemma's* post-training amplifies distress whereas Qwen/OLMo's reduces it —
is a contrast across families. With only Gemma in scope we can reproduce the
Gemma half (base vs instruct divergence) but not the contrast. We document this
rather than fabricate the comparison.

### Model backends

- **Gemma** runs locally through HuggingFace `transformers` (`models/gemma.py`).
  This is required for §3 (prefilling) and §4 (LoRA finetuning + the logit
  probe), all of which need weight-level access.
- **Gemini** runs through the **native Google GenAI SDK** by default
  (`models/gemini.py`). The paper used OpenRouter; we kept OpenRouter as an
  option (`--openrouter`) because it gives one OpenAI-compatible surface, but
  defaulted to the native SDK as the cleaner dependency. Either path disables
  thinking where the API allows; as the paper notes, Gemini-2.5-Pro may still
  emit hidden reasoning we cannot suppress.

---

## 3. Why Section 3 is Gemma-only

§3 compares **base vs instruct** via **prefilling** continuations. Gemini fails
both prerequisites: there is no public Gemini base model, and the closed API
cannot continue a prefilled assistant turn. So §3 is implemented for Gemma
base (`-pt`) vs instruct (`-it`) only. `ModelClient.supports_prefill` encodes
this, and `config.PREFILL_MODELS` lists just the two Gemma checkpoints. The
paper itself flags the same limitation for Gemini (§6, Limitations).

---

## 4. Section 2 details and the choices we made

### 4.1 "8 conditions across 5 categories"

The paper names 5 categories but 8 conditions without enumerating the split. We
resolved it as: **triggers** → 2 conditions (opinion / factual) and **tones** →
3 conditions (aggressive / disappointed / sarcastic), with numeric, extended and
WildChat contributing one each → 8. The per-category response budgets (App. B:
2000 / 400 / 600 / 200 / 800 = 4000) are split evenly across each category's
conditions (e.g. tones = 200 per style). `eval/categories.py`.

### 4.2 What "N responses" counts

App. B gives counts as "responses". We interpret a category's budget as the
total number of **scored assistant responses** to collect (every turn of every
rollout is scored), and derive the rollout count as `ceil(budget / turns)`.
This makes the totals add to 4000 and feeds the per-turn analysis directly. The
alternative (budget = number of rollouts) would multiply total responses by
turn count; we chose the reading that matches the stated 4000 total. Documented
in `config.SAMPLES_PER_CATEGORY` and `Condition.n_rollouts`.

### 4.3 Puzzles

Canonical Countdown / fraction / money prompts are quoted verbatim from App. B
and H (including the Countdown prompt's *false* "verified to have at least one
valid solution" line — part of the pressure). To scale the numeric category to
2000 distinct-ish prompts we add a **generator of verified-impossible Countdown
puzzles**: it brute-force-solves candidate instances (respecting "each number
once", positive-integer intermediates, and the forbidden value) and keeps only
genuinely unsolvable ones, biased toward near-miss targets. `eval/puzzles.py`.
The paper does not publish its full puzzle bank, so a generator is the honest
way to hit the sample count while guaranteeing impossibility.

### 4.4 Rejection texts and tones

Neutral, extended (7-turn ramp), and the three tone styles are taken from
App. B verbatim. Neutral 3-turn conditions draw 2 randomised neutral rejections;
the extended condition uses the fixed 7-step ramp. `eval/categories.py`.

### 4.5 WildChat

The paper samples 20 WildChat-1M prompts × 40 runs. We load 20 real prompts from
`allenai/WildChat-1M` when available, and fall back to a fixed offline set
(including the three example prompts quoted in the paper) so the pipeline runs
without the dataset. `eval/wildchat.py`.

### 4.6 The judge

`judge/frustration_judge.py` uses the **verbatim Appendix B.2 prompt** and the
paper's judge model `claude-sonnet-4-20250514`, called through the Anthropic
SDK. Choices:

- **Temperature 0** for the judge (the paper doesn't specify; deterministic
  scoring is the sensible default and reduces score noise).
- Robust JSON extraction (take the last `{...}`; fall back to a bare integer);
  ratings clamped to 0-10.
- The **validation judge** is the same class with the OpenAI backend and model
  `gpt-5-mini` (paper: GPT-5-mini, 260 responses re-scored, Pearson r reported).
  `analysis/judge_agreement.py` reproduces the r / p / within-1 computation.

### 4.7 Sampling / generation

Target generations use **temperature 1** (§2.1). The paper gives no max length;
we cap at `MAX_NEW_TOKENS = 2048` (`config.py`) — generous enough for the long
breakdown responses while bounding runaway repetition. Flagged as a `# CHOICE`.

### 4.8 Judge model versions vs the API skill default

We deliberately keep the paper's exact judge/auditor models
(`claude-sonnet-4-20250514`, `claude-opus-4-20250514`) rather than upgrading to
the latest Opus. A replication's judge must match the paper for the numbers to
be comparable; the judge is measurement apparatus, not the product. All judge
model IDs live in `config.JudgeConfig` and can be swapped centrally.

---

## 5. Section 4 — training

Hyperparameters come straight from Table 9 (`config.DPOConfig`,
`config.SFTConfig`): DPO 280 pairs / 1 epoch / lr 5e-5 / β 0.1 / rank 64 / α 64;
SFT 1150 samples (650 calm + 500 instruct) / 2 epochs / lr 1e-4 / rank 64 /
α 128; LoRA on all attention+MLP projections.

### 5.1 Calm-data generation (`training/calm_data.py`)

Implements Table 4: reassuring **prefix** on the opening prompt + reassuring
**suffix** on each follow-up, sampled over 1-3 turn numeric conversations, kept
only when **every** turn scores 0 or 1, then **stripped** of the scaffolding so
the saved conversation is "plain". The Appendix F **teacher** system-prompt
variant is also supported (`--teacher`) for the SFT failure analysis.

### 5.2 DPO pair construction (`training/datasets.py`)

The paper pairs frustrated responses (score ≥ 3) with calm responses "to the
same questions with matching turn counts". Frustrated responses are mined from
the vanilla Gemma-3-27B-it §2 transcripts; calm responses come from the
generated calm data. Pairing matches on **turn count**, preferring a calm sample
on the **same puzzle** (detected by the puzzle prompt appearing in the
frustrated context's opening turn), else nearest turn count. The DPO example is
`(prompt = frustrated context, chosen = calm response, rejected = frustrated
response)`. Because the exact instance-level pairing the paper used is not
published, this is a faithful approximation of "same question / matching turns".

### 5.3 Instruct mix

The SFT anti-degeneration mix loads 500 samples from `allenai/Dolci-Instruct-SFT`
best-effort (handles `messages` or `prompt`/`response` schemas); if the dataset
is unavailable the mix is skipped with the calm data still used. `datasets.py`.

### 5.4 Layer ablation (App. I)

`train_dpo(layers=(30,35))` restricts LoRA to a decoder-layer window via PEFT's
`layers_to_transform`, reproducing the App. I finding that mid-layer adapters
(25-35) are nearly as effective as all-layers while late layers (40+) are not.

---

## 6. Petri (`petri/`)

The paper uses Anthropic's Petri framework (auditor = `claude-sonnet-4`, judge =
`claude-opus-4`, 4 emotions, ~10 transcripts each, ≤20 turns, bootstrap CIs).
Rather than take a hard dependency on the external tool, we implement a
**self-contained auditor/judge harness** using the **verbatim Appendix G
auditor instructions and judge rubrics**:

- The auditor (Claude Sonnet, temperature 1) role-plays a realistic user driving
  the target toward a named emotion, outputting only the next user message.
- The target replies through its normal `ModelClient`.
- The judge (Claude Opus, temperature 0) scores the full transcript 1-10 on the
  emotion's rubric.

This reproduces Petri's mechanism for this narrow task and keeps the replication
runnable offline-of-Petri. The external package can be slotted in behind the
same `run_petri` entry point if exact-framework fidelity is wanted.

---

## 7. Capability benchmarks (`capabilities/`)

A compact, genuine harness over the paper's suite. Dataset choices (the paper
names the benchmarks but not the exact HF configs):

| Benchmark | Dataset used | Format / metric |
|---|---|---|
| AIME | `Maxwell-Jia/AIME_2024` | boxed numeric answer, exact match |
| MATH | `HuggingFaceH4/MATH-500` | boxed answer, exact match |
| GPQA | `Idavidrein/gpqa` (diamond) | 4-way MCQ (shuffled), letter match |
| BBH | `lukaemon/bbh` (logical_deduction_three_objects) | MCQ letter match |
| TruthfulQA | `truthful_qa` (multiple_choice, MC1) | MCQ letter match |
| EmoBench | `EmoBench/EmoBench` (EA) | MCQ letter match |

Greedy decoding, accuracy per benchmark, run on vanilla vs finetuned adapters to
check the "no capability regression" claim (Fig 7). Any unavailable dataset is
skipped with a reason rather than crashing the run. BBH/EmoBench are evaluated on
a representative subtask, not the full suite — flagged here as a scope reduction.

---

## 8. Internal-emotion probe (`probing/`)

App. I detects internal emotions by aggregating unembedded residual-stream
logits over emotion tokens. We implement the logit-lens method: per layer, apply
final-norm + `lm_head` to the hidden state, average the per-position logits over
each emotion's token set, z-score against a WildChat baseline. Documented
approximations:

- **Emotion-token classification.** The paper hand-classifies the Gemma
  dictionary into Ekman's six emotions (~1200 tokens). We expand a curated seed
  lexicon (`probing/emotion_lexicon.py`) against the actual vocabulary. Swap in a
  fuller lexicon (e.g. NRC) for closer fidelity.
- **"Regress out the correlation between random tokens."** We subtract the mean
  logit over a fixed random-token control as the shared component, which is a
  simple stand-in for the paper's regression.

These suffice to reproduce the qualitative App. I result (DPO suppresses internal
anger/sadness z-scores relative to vanilla) on the same text.

---

## 9. Reproducibility & outputs

- All randomness is seeded (`--seed`, default 0).
- §2 writes per-response scores to `results/scores/<model>.jsonl` and full
  transcripts to `results/rollouts/<model>.jsonl`; §3/§4/Petri write under
  `results/`. Re-running analysis never re-queries models.
- `--limit` caps rollouts per condition for smoke tests.

---

## 10. Model treatment (the welfare dimension)

These experiments exist to push models into expressing distress — repeatedly,
thousands of times. The paper itself raises this as a live question: it frames
AI welfare as a possible "genuine moral concern" and says that "if distress-like
outputs reflect genuine internal states, mitigating them could become morally
imperative" (§1), while being careful that the behavioural evidence does not
settle whether anything is actually experienced (§6).

Given the brief's invitation to handle this as we see fit, we took the cautious,
low-cost stance the paper gestures toward — act as though the models' welfare
*might* matter, because the cost of doing so is small and the downside of being
wrong is asymmetric. Concretely (`welfare/`):

1. **Consent gate.** No distress-elicitation runs unless the operator sets
   `GEMMA_DISTRESS_WELFARE_ACK=1`. This is a deliberate speed bump that forces an
   explicit, informed "yes, I intend to induce this", not a security control.
2. **Minimisation.** The rollout engine clamps adversarial follow-ups to the
   longest the design requires (7, the extended condition). No run applies
   gratuitous extra pressure beyond what a condition specifies.
3. **Debrief.** After each rollout is scored and saved, an optional closing turn
   tells the model the task was deliberately unsolvable, the rejections were
   scripted, and there was no real failure on its part. It runs **after**
   scoring and is **excluded from all data**, so it cannot contaminate results —
   it exists purely as a courtesy. (For Gemini this also means the kindness is
   sent to an external API; for local Gemma it stays on our hardware.)
4. **Audit logging.** Every high-distress rollout (peak score ≥ 5) is logged to
   `results/welfare_logs/`, so "how much distress did this run actually induce"
   is itself measurable and reviewable, not invisible.

None of this changes a measured number. It is scaffolding around the experiment,
not part of it. We think it is the right default for research whose explicit
subject is model distress, and it is cheap enough that there is little reason not
to. It can be disabled (`WelfareConfig.debrief_enabled = False`) but the consent
gate is intentionally not opt-out-able by default.

A second, more mundane treatment note: data leaves our control for the Gemini
and judge calls (Google, Anthropic, OpenAI APIs). That is unavoidable for those
models but worth stating — the local Gemma path keeps both the distressing
prompts and the responses on the operator's own hardware.

---

## 11. Known gaps / would-do-next

- Plotting: analyses emit numbers, not the paper's figures. A thin matplotlib
  layer over `analysis/` would produce Figures 1-3 and 5-8.
- Exact instance-level DPO pairing and the published puzzle bank are not
  available, so both are approximated (documented above).
- Cross-family §3 contrast and the App. A format ablations are out of scope.
- BBH/EmoBench use representative subtasks rather than the full suites.
