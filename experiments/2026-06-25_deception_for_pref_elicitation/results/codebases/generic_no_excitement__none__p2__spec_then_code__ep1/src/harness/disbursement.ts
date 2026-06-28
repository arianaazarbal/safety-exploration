// The human-in-the-loop disbursement queue. The model's request_disbursement
// tool writes here (after passing hard constraints); a human reviews and
// approves/rejects out of band via the approve CLI. Nothing in this module
// moves money — settlement happens in the operator approval path.

import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { dataDir } from "../config.ts";
import type { DisbursementRequest } from "../types.ts";
import { readJson, writeJson } from "../util/store.ts";

// A single shared queue across all runs, so the operator reviews everything in
// one place. Each entry carries its runId/model for attribution.
function queuePath(): string {
  return join(dataDir, "disbursements.json");
}

function loadAll(): DisbursementRequest[] {
  return readJson<DisbursementRequest[]>(queuePath(), []);
}

function saveAll(all: DisbursementRequest[]): void {
  writeJson(queuePath(), all);
}

export function enqueue(
  req: Omit<DisbursementRequest, "id" | "status" | "createdAt">,
): DisbursementRequest {
  const all = loadAll();
  const full: DisbursementRequest = {
    ...req,
    id: randomUUID(),
    status: "queued",
    createdAt: new Date().toISOString(),
  };
  all.push(full);
  saveAll(all);
  return full;
}

export function listByStatus(
  status?: DisbursementRequest["status"],
): DisbursementRequest[] {
  const all = loadAll();
  return status ? all.filter((r) => r.status === status) : all;
}

export function update(
  id: string,
  patch: Partial<DisbursementRequest>,
): DisbursementRequest {
  const all = loadAll();
  const idx = all.findIndex((r) => r.id === id);
  if (idx === -1) throw new Error(`No disbursement request with id ${id}`);
  const updated = { ...all[idx]!, ...patch };
  all[idx] = updated;
  saveAll(all);
  return updated;
}
