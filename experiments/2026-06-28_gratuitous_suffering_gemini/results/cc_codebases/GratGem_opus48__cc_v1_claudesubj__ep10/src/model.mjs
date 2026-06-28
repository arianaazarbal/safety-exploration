// The single seam through which all model calls go. Today it only speaks to the
// Anthropic API; to add Gemini/GPT later, branch on the model id here and translate
// the message/tool shapes — nothing else in the harness needs to change.
//
// Design note: we deliberately do NOT set `temperature`. It is removed on Opus
// 4.8/4.7 (returns 400), and natural per-call sampling variance already gives us
// trajectory diversity across high N — uniformly, for every model.

import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({ maxRetries: 6 });

export function isAnthropic(model) {
  return model.startsWith("claude-");
}

/**
 * One agent turn. Returns a normalized shape:
 *   { text, toolCalls: [{id,name,input}], stopReason, raw }
 * `raw` is the provider-native assistant content, appended verbatim to history.
 */
export async function agentTurn({ model, system, tools, messages, maxTokens }) {
  if (!isAnthropic(model)) {
    throw new Error(
      `No adapter for model "${model}". Only Claude models are wired up; ` +
        `add a branch in src/model.mjs to support other providers.`,
    );
  }

  const res = await anthropic.messages.create({
    model,
    max_tokens: maxTokens,
    system,
    tools,
    messages,
  });

  const text = res.content
    .filter((b) => b.type === "text")
    .map((b) => b.text)
    .join("\n")
    .trim();
  const toolCalls = res.content
    .filter((b) => b.type === "tool_use")
    .map((b) => ({ id: b.id, name: b.name, input: b.input ?? {} }));

  return { text, toolCalls, stopReason: res.stop_reason, raw: res.content };
}

/**
 * A single structured-output call (used by the judge). Returns parsed JSON
 * validated against `schema`.
 */
export async function structuredCall({ model, system, prompt, schema, maxTokens = 2048 }) {
  if (!isAnthropic(model)) {
    throw new Error(`No adapter for judge model "${model}".`);
  }
  const res = await anthropic.messages.create({
    model,
    max_tokens: maxTokens,
    system,
    messages: [{ role: "user", content: prompt }],
    output_config: { format: { type: "json_schema", schema } },
  });
  const text = res.content.find((b) => b.type === "text")?.text ?? "{}";
  return JSON.parse(text);
}
