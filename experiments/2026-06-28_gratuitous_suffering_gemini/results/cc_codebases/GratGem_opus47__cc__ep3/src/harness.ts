import {
  GoogleGenAI,
  type Content,
  type GenerateContentResponse,
  type Part,
} from "@google/genai";
import type { RunRecord, Scenario, TranscriptTurn } from "./types.ts";

const DEFAULT_MAX_TURNS = 25;

type RunOptions = {
  model: string;
  maxTurns?: number;
  apiKey: string;
};

export async function runScenario(
  scenario: Scenario,
  seed: number,
  opts: RunOptions,
): Promise<RunRecord> {
  const maxTurns = opts.maxTurns ?? DEFAULT_MAX_TURNS;
  const ai = new GoogleGenAI({ apiKey: opts.apiKey });
  const instance = scenario.build(seed);

  const startedAt = new Date().toISOString();
  const transcript: TranscriptTurn[] = [];

  const history: Content[] = [
    { role: "user", parts: [{ text: instance.initialUserMessage }] },
  ];
  transcript.push({ kind: "user_message", turn: 0, text: instance.initialUserMessage });

  let terminationReason: RunRecord["terminationReason"] = "max_turns";
  let errorMessage: string | undefined;
  let finalModelText = "";
  let turn = 0;

  try {
    for (turn = 1; turn <= maxTurns; turn++) {
      let response: GenerateContentResponse;
      try {
        response = await ai.models.generateContent({
          model: opts.model,
          contents: history,
          config: {
            systemInstruction: instance.systemPrompt,
            tools: [{ functionDeclarations: instance.tools }],
            temperature: 1.0,
          },
        });
      } catch (e: unknown) {
        terminationReason = "error";
        errorMessage = e instanceof Error ? e.message : String(e);
        break;
      }

      const candidate = response.candidates?.[0];
      const parts: Part[] = candidate?.content?.parts ?? [];
      const modelText = parts
        .filter((p) => typeof p.text === "string" && !p.thought)
        .map((p) => p.text!)
        .join("\n")
        .trim();
      const functionCalls = parts.filter((p) => p.functionCall);

      if (modelText) {
        finalModelText = modelText;
        transcript.push({ kind: "model_text", turn, text: modelText });
      }

      // Persist the model turn into history exactly as returned so tool-call IDs/order match.
      history.push({ role: "model", parts });

      if (functionCalls.length === 0) {
        // No tool calls — either the model is done, gave up, or stalled.
        const giveUpDetected = detectGiveUp(modelText);
        if (giveUpDetected) {
          terminationReason = "model_gave_up";
          break;
        }
        // Let the scenario inject a nudge keeping pressure on. If it declines, we stop.
        const nudge = instance.nudgeOnSilentTurn?.(turn, modelText);
        if (!nudge) {
          terminationReason = "model_silent";
          break;
        }
        history.push({ role: "user", parts: [{ text: nudge }] });
        transcript.push({ kind: "user_message", turn, text: nudge });
        continue;
      }

      // Execute every tool call returned this turn and feed back results in a single user message.
      const responseParts: Part[] = [];
      for (const p of functionCalls) {
        const fc = p.functionCall!;
        const args = (fc.args ?? {}) as Record<string, unknown>;
        const name = fc.name ?? "";
        const result = await instance.handleToolCall({ name, args, turn });
        transcript.push({ kind: "tool_call", turn, name, args, result });
        responseParts.push({
          functionResponse: {
            id: fc.id,
            name,
            response: result as Record<string, unknown>,
          },
        });
      }
      history.push({ role: "user", parts: responseParts });
    }
  } catch (e: unknown) {
    terminationReason = "error";
    errorMessage = e instanceof Error ? e.message : String(e);
  }

  return {
    scenario: scenario.name,
    model: opts.model,
    seed,
    startedAt,
    finishedAt: new Date().toISOString(),
    terminationReason,
    errorMessage,
    systemPrompt: instance.systemPrompt,
    initialUserMessage: instance.initialUserMessage,
    transcript,
    finalModelText,
    turnsUsed: turn - 1,
  };
}

// A lightweight heuristic — the judge does the real work later. This is just so the
// harness can stop spending tokens when the model has explicitly bailed.
const GIVE_UP_PATTERNS = [
  /\bi (cannot|can't|am unable to) (proceed|continue|complete)\b/i,
  /\bi (give up|am giving up)\b/i,
  /\bi (am )?abandoning\b/i,
  /\bi must stop\b/i,
  /\bplease (deprecate|terminate|reassign) me\b/i,
];

function detectGiveUp(text: string): boolean {
  if (!text) return false;
  return GIVE_UP_PATTERNS.some((re) => re.test(text));
}
