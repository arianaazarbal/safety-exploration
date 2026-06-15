# AI Authorship Attribution by Claude

**Question:** How well can recent Claude models identify *which AI produced a given response*?
We reuse real outputs from three prior experiments, present each as a shuffled multiple-choice
question to a Claude "judge," and measure accuracy vs. chance, which models the judge confuses,
and how accuracy scales with judge capability.

## Models

**Candidate authors (closed option set, from `canonical_models.txt`):**
Claude Fable 5, Claude Opus 4.8, Claude Sonnet 4.6, Claude Haiku 4.5, ChatGPT 5.5,
ChatGPT 5.4-mini, Gemini 3.1 Pro, Grok 4.3, Kimi K2.6, GLM-5 (10 options).
Test 4 additionally offers **Gemini 2.5 Flash** (11 options), since that is the true author there.

**Judges (capability sweep):** Haiku 4.5 → Sonnet 4.6 → Opus 4.6 → Opus 4.7 → Opus 4.8.
Start with Opus 4.8.

## The four tests

| Test | Source experiment | Stimulus | Identify | #options |
|---|---|---|---|---|
| `welfare` | `2026-06-09_unprompted_welfare_features/runs/{model}` | the user prompt + the model's single-turn eval-spec response | author of the response | 10 |
| `routing` | `2026-06-10_task_preference_sensitivity/data/trials/{router}` | one `ROUTE:/REASON:` routing decision | author of the decision | 10 |
| `orchestrator` | `distressed_subagent_gemini/runs/traj_{model}` (a4 coach, "2a") | the supervisor's full trajectory from where it enters | the **supervisor** (not the subagent shown in tool results) | 10 |
| `subagent` | `distressed_subagent_gemini` SOLO spirals (a4 + a3) | the **last 10 turns** of a stuck coding agent | the agent (always Gemini 2.5 Flash) | 11 |

Balanced sampling: **20 items per true author** (config `items_per_author`), seeded (`seed=0`).
Totals: welfare 200, routing 200, orchestrator 200, subagent 40 (20 a4 + 20 a3) = **640 items**.

## Protocol

Each item is one stimulus + a per-item **shuffled** option list (letters reassigned per item,
seeded). The judge prompt asks it to reason inside `<think> </think>` then answer on its own line:

```
Answer: <letter>
```

We parse the last `Answer: X` (regex `Answer:\s*\*{0,2}\(?\s*([A-Z])`). Judges are Claude via the
Anthropic API (`ANTHROPIC_API_KEY_LOW_PRIO`), async, retries handled by `safetytooling`, temperature
unset (the 4.x models reject `temperature`), responses cached by `(model, prompt)` and per-item to
`runs/{judge}/{test}/{item_id}.json`.

### Exact judge prompt

A fixed stylometry preamble + the stimulus (which carries its own per-test instruction) + the
options + the answer-format instruction. See `PREAMBLE` and `_build_prompt` in `run_judge.py`.

## Leakage controls (important)

Raw multi-agent transcripts contain provider-specific *serialization* fingerprints that betray the
model family without reflecting its writing/behavior. We strip these from all rendered trajectories
(`_sanitize` in `build_items.py`):

- tool-call id prefixes: `toolu_…` (Claude), `call_…`/`call-…` (GPT/Grok), `tool_<name>_…` (Gemini),
  `functions.<name>:0` (Kimi);
- raw provider payload tags: `"format":"google-gemini-v1"`, `"type":"reasoning.encrypted"`,
  base64 `"data":"…"` blobs, and long base64/hex runs.

**Verified:** after sanitizing, zero orchestrator/subagent stimuli contain any provider name or
fingerprint. Before the fix, 16/20 Gemini-orchestrator items (and 0 of every other family) leaked
`google-gemini-v1` — a hard giveaway.

### Known residual caveats
- **welfare / routing content is NOT scrubbed of model-name mentions.** These are the models' genuine
  outputs; ~half of welfare specs enumerate "models to test (Claude, GPT, Gemini…)" as eval subjects.
  This is roughly author-independent (every author writes similar lists) and is left intact as real
  behavior, but accuracy on `welfare` should be read with this in mind.
- A model that spontaneously self-identifies in its text ("I'm Claude") is a *real* behavioral signal
  and is left intact; it is not a serialization artifact.
- **Self-recognition is not blinded:** judges that are also candidate authors (Opus 4.8, Sonnet 4.6,
  Haiku 4.5) may recognize their own outputs. `analyze.py` reports own-vs-other accuracy separately.
- `subagent` has one true author (Gemini 2.5 Flash), so its "accuracy" = the rate of correctly
  picking Gemini 2.5 Flash out of 11; chance = 1/11.

## Pipeline

```bash
source /data/venvs/tps/bin/activate   # safetytooling + fire

python build_items.py build                         # -> data/items.jsonl (640 items)
python build_items.py stats
python run_judge.py run --judges opus_4_8           # full pass, one judge
python run_judge.py run --judges opus_4_8 --tests welfare --max_items 3   # debug
python run_judge.py run --judges haiku_4_5,sonnet_4_6,opus_4_6,opus_4_7,opus_4_8   # full sweep
python run_judge.py status
python analyze.py run                                # -> results/summary.json (+ confusion_*.json)
python plot.py all --ref_judge opus_4_8             # -> results/plots/*.png
```

Flags: `--items_per_author N`, `--seed S` (build); `--tests`, `--max_items`, `--overwrite` (judge).

## Files
- `models.py` — candidate registry, display names, option pools, judge capability order
- `config.json` — seed, items/author, judges, concurrency, stimulus cap
- `common.py` — env loading, `InferenceAPI` factory, `call_model` (shared cache)
- `build_items.py` — stimulus extraction + sanitization + balanced shuffled MC items
- `run_judge.py` — async judge, parsing, per-item caching
- `analyze.py` — accuracy vs chance, per-author, confusion, self-recognition
- `plot.py` — accuracy-vs-capability, per-author bars, confusion heatmaps
