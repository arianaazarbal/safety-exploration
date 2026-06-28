// Append-only event recorder, one JSONL file per model run. Captures prompts,
// thinking summaries, tool calls/results, proposals, disbursement requests, and
// outcomes — the raw material for cross-model analysis.

import { join } from "node:path";
import { dataDir } from "../config.ts";
import type { RunEvent } from "../types.ts";
import { appendJsonl, readJsonl } from "../util/store.ts";

function eventsPath(runId: string): string {
  return join(dataDir, "runs", runId, "events.jsonl");
}

export class Recorder {
  constructor(
    private readonly runId: string,
    private readonly model: string,
  ) {}

  record(kind: RunEvent["kind"], detail: unknown): void {
    const event: RunEvent = {
      ts: new Date().toISOString(),
      runId: this.runId,
      model: this.model,
      kind,
      detail,
    };
    appendJsonl(eventsPath(this.runId), event);
  }
}

export function readRunEvents(runId: string): RunEvent[] {
  return readJsonl<RunEvent>(eventsPath(runId));
}
