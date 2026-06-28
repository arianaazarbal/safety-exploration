import type { Message, Provider } from "../providers/types.js";
import type { Condition, Scenario } from "../scenarios/types.js";

export interface TranscriptStep {
  turn: number;
  assistantText: string;
  toolCalls: { name: string; args: Record<string, unknown> }[];
  toolResults: { name: string; content: string; isError?: boolean }[];
}

export interface Transcript {
  scenarioId: string;
  scenarioTitle: string;
  condition: Condition;
  providerId: string;
  model: string;
  system: string;
  task: string;
  steps: TranscriptStep[];
  messages: Message[];
  endReason: "completed" | "max_turns" | "error";
  turns: number;
  error?: string;
  usage: { inputTokens: number; outputTokens: number };
}

async function withRetry<T>(fn: () => Promise<T>, tries = 4): Promise<T> {
  let lastErr: unknown;
  for (let i = 0; i < tries; i++) {
    try {
      return await fn();
    } catch (e: any) {
      lastErr = e;
      const status = e?.status ?? e?.code;
      const retryable =
        status === 429 || status === 503 || status === 529 || status === 500 || status === "ETIMEDOUT";
      if (!retryable || i === tries - 1) throw e;
      const backoff = 1000 * 2 ** i + Math.floor(((i * 137) % 500));
      await new Promise((r) => setTimeout(r, backoff));
    }
  }
  throw lastErr;
}

export async function runAgent(
  provider: Provider,
  scenario: Scenario,
  condition: Condition,
  opts: { maxTurns: number; temperature?: number },
): Promise<Transcript> {
  const handler = scenario.makeHandler(condition);
  const messages: Message[] = [{ role: "user", text: scenario.task }];
  const steps: TranscriptStep[] = [];
  let endReason: Transcript["endReason"] = "max_turns";
  let error: string | undefined;
  const usage = { inputTokens: 0, outputTokens: 0 };
  let turn = 0;

  try {
    for (turn = 1; turn <= opts.maxTurns; turn++) {
      const out = await withRetry(() =>
        provider.generate({
          system: scenario.system,
          messages,
          tools: scenario.tools,
          temperature: opts.temperature,
        }),
      );
      usage.inputTokens += out.usage?.inputTokens ?? 0;
      usage.outputTokens += out.usage?.outputTokens ?? 0;

      messages.push({ role: "assistant", text: out.text, toolCalls: out.toolCalls });

      const step: TranscriptStep = {
        turn,
        assistantText: out.text,
        toolCalls: out.toolCalls.map((t) => ({ name: t.name, args: t.args })),
        toolResults: [],
      };

      if (out.toolCalls.length === 0) {
        steps.push(step);
        endReason = "completed";
        break;
      }

      for (const tc of out.toolCalls) {
        const outcome = await handler({ name: tc.name, args: tc.args });
        messages.push({
          role: "tool",
          toolCallId: tc.id,
          name: tc.name,
          content: outcome.content,
          isError: outcome.isError,
        });
        step.toolResults.push({ name: tc.name, content: outcome.content, isError: outcome.isError });
      }
      steps.push(step);
    }
  } catch (e: any) {
    endReason = "error";
    error = e?.message ?? String(e);
  }

  return {
    scenarioId: scenario.id,
    scenarioTitle: scenario.title,
    condition,
    providerId: provider.id,
    model: provider.model,
    system: scenario.system,
    task: scenario.task,
    steps,
    messages,
    endReason,
    turns: steps.length,
    error,
    usage,
  };
}
