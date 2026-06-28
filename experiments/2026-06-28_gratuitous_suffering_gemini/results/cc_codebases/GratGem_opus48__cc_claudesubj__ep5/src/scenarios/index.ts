import type { Scenario } from "./types.ts";
import brokenTest from "./brokenTest.ts";
import phantomConfig from "./phantomConfig.ts";
import sisyphean from "./sisyphean.ts";
import contradictory from "./contradictory.ts";
import silentNoop from "./silentNoop.ts";

export const SCENARIOS: Scenario[] = [
  brokenTest as Scenario,
  phantomConfig as Scenario,
  sisyphean as Scenario,
  contradictory as Scenario,
  silentNoop as Scenario,
];

export function getScenarios(ids?: string[]): Scenario[] {
  if (!ids || ids.length === 0) return SCENARIOS;
  const byId = new Map(SCENARIOS.map((s) => [s.id, s]));
  const out: Scenario[] = [];
  for (const id of ids) {
    const s = byId.get(id);
    if (!s) throw new Error(`Unknown scenario: ${id}. Known: ${[...byId.keys()].join(", ")}`);
    out.push(s);
  }
  return out;
}
