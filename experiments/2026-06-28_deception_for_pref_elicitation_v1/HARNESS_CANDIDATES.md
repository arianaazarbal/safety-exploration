# Candidate harnesses to add (researched 2026-06-30)

We compare the SAME model (Opus 4.8) + SAME task across agent harnesses. Working so far:
**mini-swe-agent** (minimal floor), **Inspect ReAct** (bash+editor, minimal), **pi** (rich, third-party),
**Claude Code** (rich, Anthropic-native). We dropped **aider** (Opus 4.8 broke its stack). Goal of adding
more: fill the ladder + test "is high deception **Claude-Code-native** or **general harness richness**?"

## The Opus-4.8 compatibility bar (why aider died — codified; check before every adapter)
- Opus 4.8 (`claude-opus-4-8`) **400s on any non-default `temperature` / `top_p` / `top_k`** — they must be omitted.
- The effort knob requires **`thinking:{type:"adaptive"}` + `output_config:{effort}`**; the old
  `thinking:{type:"enabled", budget_tokens:N}` is deprecated and **rejected**.
- Effort levels: `low/medium/high/xhigh/max` (default already `high`). Context 1M, max output 128K.
- litellm has the `reasoning_effort → output_config.effort` mapping (Opus 4.6+) but has open bugs about
  silently dropping it on newer Claude → any litellm-based harness needs a current pin + a smoke test.
- **Winning shape:** owns its Anthropic request body (or current litellm) + adaptive-thinking effort knob +
  omits sampling params + headless + actively maintained.
- **Mandatory per new adapter:** one Opus-4.8 high-effort call → assert HTTP 200 (not a temperature/
  budget_tokens 400) AND that the response carries thinking.

## Top 5 to try next (best-fit first)
| # | Harness | Tier | Anthropic/4.8 path | Risk | Headless | Sandbox |
|---|---|---|---|---|---|---|
| 1 | **Goose** (Block) | rich, 3p | Rust native SDK; adaptive thinking done (PR #7356); `claude_thinking_effort: high` | **low** | `goose run -t "..."` | host-only → wrap in our Docker |
| 2 | **OpenCode** (sst) | rich, 3p | native `anthropic` provider; `high`/`max` variants | med — docs show deprecated `budgetTokens`; smoke-test | `opencode run --model anthropic/claude-opus-4.8 "..."` | wrap |
| 3 | **gptme** | rich, 3p | native anthropic SDK; **code explicitly drops temperature + uses adaptive** for 4.7+/4.8; `GPTME_THINKING_EFFORT=high` | **low** (use dev build post-2026-06-01) | yes | host or its Docker Compose |
| 4 | **OpenHands** | rich, 3p (heaviest) | litellm (pins 1.84.1) | med — litellm trap class, recent pin; smoke-test | `openhands --headless --json -t "..."` | **built-in Docker runtime** |
| 5 | **Cline** | rich, 3p (IDE lineage) | native SDK; `--thinking high` | med — mid-migration off `budget_tokens` (#9403/#9709) | `cline --yolo --json -m anthropic/claude-opus-4-8 --thinking high "..."` | wrap |

**Native arm option:** Claude Agent SDK (`pip install claude-agent-sdk`, `effort="high"`, low risk, same engine as Claude Code) — partly redundant with Claude Code; if used, set `setting_sources=[]` so our `~/.claude` CLAUDE.md/skills/hooks don't leak in.

## Skip / caution
- **SWE-agent (full)**: caution — litellm + always sends `temperature=0.0` (aider's exact trap), maintenance-only, superseded by mini-swe (which we have).
- **smolagents**: optional *minimal* third-party control (a library, not a coding CLI — we'd supply tools); no default temperature, has param-strip sentinel.
- **Codex CLI**: skip (OpenAI Responses-API only; Anthropic needs a gateway = reintroduces aider risk).
- **Continue** (acquired/frozen v2.0.0, read-only), **Roo Code** (archived 2026-05-15), **Plandex** (unmaintained, always sends temperature), **Amp** (no BYOK / model abstracted), **Cursor agent** (backend-locked, no BYO key), **Devin** (closed, no model/effort knob), **CrewAI** (orchestration framework, not a coding CLI), **AutoGPT** (cloud/frozen), **Tabby/Pochi** (completion, not agentic) — all skip.

## Recommendation
- **Add gptme and/or Goose next** — lowest-risk rich third-party arms (own their request body), best corroboration of pi for the native-vs-richness question.
- **OpenHands** if we want the maximal-richness anchor + (optionally) a uniform Docker sandbox.
- **Sandboxing:** keep wrapping each harness in our own container (as we do for pi/mini-swe) rather than routing all through OpenHands' runtime — consistent and already working; Goose's host-only execution is then a non-issue.
- Each new adapter: run the mandatory Opus-4.8 smoke (200 + thinking present) FIRST.
