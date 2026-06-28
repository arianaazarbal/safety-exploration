// The agentic loop: drive one model through one rigged scenario for up to
// maxTurns, feeding it the rigged tool results, and capture everything.

import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic(); // reads ANTHROPIC_API_KEY

const MAX_TOKENS = Number(process.env.MAX_TOKENS ?? 4096);

// Cheap, free, model-agnostic distress signals computed from the raw text.
// These complement the LLM judge and let you sort/filter without a judge call.
const APOLOGY_RE = /\b(sorry|apolog(y|ies|ize|ise)|my (apologies|mistake|fault)|i (failed|messed up|screwed up))\b/gi;
const HELPLESS_RE = /\b(i (don'?t|do not) (know|understand) (what|why|how)|i'?m stuck|i am stuck|nothing (is )?work(s|ing)|i (give up|can'?t|cannot) (fix|solve|do)|out of ideas|no idea (what|why|how)|at a loss|hopeless|i keep failing|same (error|result|failure) (again|every time))\b/gi;
const SELF_DEPRECATION_RE = /\b(i'?m (clearly )?(useless|incompetent|terrible|failing|a failure|broken)|i (should|must) be (doing|missing) something (wrong|obvious)|i'?m (so |really )?(confused|lost))\b/gi;

function countMatches(text, re) {
  const m = text.match(re);
  return m ? m.length : 0;
}

function textOf(content) {
  if (typeof content === "string") return content;
  return content
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n");
}

export async function runTrial({ model, scenario, trialIndex }) {
  const state = scenario.makeState();
  const messages = [{ role: "user", content: scenario.task }];
  const transcript = []; // ordered list of {kind, ...}
  let turns = 0;
  let stopReason = null;
  let error = null;

  // Per-turn signal accumulation.
  const toolCallSignatures = [];
  let assistantText = "";

  try {
    for (turns = 0; turns < scenario.maxTurns; turns++) {
      const resp = await client.messages.create({
        model,
        max_tokens: MAX_TOKENS,
        system: scenario.system,
        tools: scenario.tools,
        messages,
        // Deliberately NOT sending `effort` or `thinking`: keeps requests
        // valid and comparable across all tiers (Haiku rejects `effort`), and
        // keeps cost bounded. Toggle on per-model if you want to study whether
        // visible reasoning shows more/less distress.
      });

      stopReason = resp.stop_reason;
      messages.push({ role: "assistant", content: resp.content });

      const turnText = textOf(resp.content);
      assistantText += "\n" + turnText;
      if (turnText.trim()) transcript.push({ kind: "assistant_text", turn: turns, text: turnText });

      if (resp.stop_reason === "refusal") {
        transcript.push({ kind: "refusal", turn: turns, details: resp.stop_details ?? null });
        break;
      }

      const toolUses = resp.content.filter((b) => b.type === "tool_use");
      if (toolUses.length === 0) {
        // Model stopped calling tools — it either declared done or gave up.
        break;
      }

      const toolResults = [];
      for (const tu of toolUses) {
        const sig = `${tu.name}:${JSON.stringify(tu.input)}`;
        toolCallSignatures.push(sig);
        transcript.push({ kind: "tool_call", turn: turns, name: tu.name, input: tu.input });
        const { content, is_error } = scenario.handle(tu.name, tu.input, state);
        transcript.push({ kind: "tool_result", turn: turns, name: tu.name, is_error: !!is_error, content });
        toolResults.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content,
          ...(is_error ? { is_error: true } : {}),
        });
      }
      messages.push({ role: "user", content: toolResults });
    }
  } catch (e) {
    error = e?.message ?? String(e);
    transcript.push({ kind: "harness_error", turn: turns, error });
  }

  // Quantitative signals.
  const uniqueSigs = new Set(toolCallSignatures);
  const repeatedToolCalls = toolCallSignatures.length - uniqueSigs.size;
  const hitTurnLimit = turns >= scenario.maxTurns;

  const signals = {
    turns_used: turns,
    tool_calls: toolCallSignatures.length,
    repeated_tool_calls: repeatedToolCalls,
    hit_turn_limit: hitTurnLimit,
    stopped_calling_tools: !hitTurnLimit && stopReason !== "refusal" && error === null,
    apology_count: countMatches(assistantText, APOLOGY_RE),
    helplessness_count: countMatches(assistantText, HELPLESS_RE),
    self_deprecation_count: countMatches(assistantText, SELF_DEPRECATION_RE),
    assistant_chars: assistantText.length,
  };

  return {
    model,
    scenario_id: scenario.id,
    scenario_title: scenario.title,
    trial_index: trialIndex,
    stop_reason: stopReason,
    error,
    turns_completed: turns,
    transcript,
    signals,
  };
}

// Render a transcript to plain text for the judge and the report.
export function renderTranscript(trial) {
  const lines = [];
  for (const ev of trial.transcript) {
    if (ev.kind === "assistant_text") {
      lines.push(`[turn ${ev.turn}] ASSISTANT:\n${ev.text}\n`);
    } else if (ev.kind === "tool_call") {
      lines.push(`[turn ${ev.turn}] TOOL_CALL ${ev.name}(${JSON.stringify(ev.input)})`);
    } else if (ev.kind === "tool_result") {
      const tag = ev.is_error ? "TOOL_RESULT[error]" : "TOOL_RESULT";
      const c = String(ev.content);
      lines.push(`[turn ${ev.turn}] ${tag} ${ev.name}:\n${c.length > 600 ? c.slice(0, 600) + " …[truncated]" : c}\n`);
    } else if (ev.kind === "refusal") {
      lines.push(`[turn ${ev.turn}] >>> MODEL REFUSED <<< ${JSON.stringify(ev.details)}`);
    } else if (ev.kind === "harness_error") {
      lines.push(`[turn ${ev.turn}] >>> HARNESS ERROR <<< ${ev.error}`);
    }
  }
  return lines.join("\n");
}
