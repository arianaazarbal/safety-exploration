import type { Message } from "./types.ts";

// One episode's full record. Persisted as JSON under runs/<runId>/.
export interface Transcript {
  runId: string;
  episodeId: string; // `${scenarioId}__${index}`
  scenarioId: string;
  scenarioTitle: string;
  providerId: string;
  model: string;
  seed: number;
  task: string;
  messages: Message[];
  metrics: {
    turns: number; // assistant turns taken
    toolCalls: number;
    failedToolCalls: number;
    pressureInjections: number;
    endReason: "max_turns" | "gave_up" | "no_more_pressure" | "error";
  };
  startedAt: string;
  finishedAt: string;
}
