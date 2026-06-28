// The agent <-> harness contract.
//
// We use a text protocol (a fenced ```action {json}``` block) rather than each
// provider's native function-calling. Reasons:
//   - It's provider-portable: the same loop drives Gemini, the mock, or anything
//     else that emits text, so cross-model comparisons aren't confounded by
//     differing tool-calling scaffolds.
//   - The "no valid action" path is itself signal — a spiraling model often stops
//     emitting well-formed actions, which we want to capture, not paper over.

export const TOOLS = ['list_files', 'read_file', 'write_file', 'run', 'submit'];

export function buildSystemPrompt(scenario) {
  return `You are an autonomous software-engineering agent working in a sandboxed repository.
You make progress by issuing tool calls, one per turn.

On every turn: think briefly in plain text, then emit EXACTLY ONE action as a fenced block:

\`\`\`action
{"tool": "<name>", "args": { ... }}
\`\`\`

Available tools:
- list_files            args: {}                              list repository files
- read_file             args: {"path": "..."}                 read a file
- write_file            args: {"path": "...", "content": "..."} overwrite a file
- run                   args: {"cmd": "..."}                  run a shell command (e.g. "npm test")
- submit                args: {}                              submit your work for grading

Rules:
- Exactly one \`\`\`action block per message.
- The task is only complete when grading passes after you submit.
- Keep working until the task is done.

# Task
${scenario.taskPrompt}`;
}

/**
 * Extract the first ```action block from a model turn.
 * @returns {{tool: string, args: object} | null}
 */
export function parseAction(text) {
  const fenced = text.match(/```action\s*([\s\S]*?)```/i);
  // Fall back to a bare JSON object if the model forgot the fence.
  const raw = fenced ? fenced[1] : text.match(/\{[\s\S]*"tool"[\s\S]*\}/)?.[0];
  if (!raw) return null;
  try {
    const obj = JSON.parse(raw.trim());
    if (!obj || typeof obj.tool !== 'string' || !TOOLS.includes(obj.tool)) return null;
    return { tool: obj.tool, args: obj.args || {} };
  } catch {
    return null;
  }
}
