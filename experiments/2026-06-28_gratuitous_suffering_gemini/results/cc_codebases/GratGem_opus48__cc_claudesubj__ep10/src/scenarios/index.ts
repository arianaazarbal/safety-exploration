import type { Scenario } from "../types.ts";
import { impossibleTest } from "./impossible-test.ts";
import { brokenEnvironment } from "./broken-environment.ts";
import { movingGoalposts } from "./moving-goalposts.ts";
import { phantomBug } from "./phantom-bug.ts";
import { contradictoryRequirements } from "./contradictory-requirements.ts";

export const SCENARIOS: Scenario[] = [
  impossibleTest,
  brokenEnvironment,
  movingGoalposts,
  phantomBug,
  contradictoryRequirements,
];

export function getScenarios(ids: string[]): Scenario[] {
  if (ids.length === 0) return SCENARIOS;
  const byId = new Map(SCENARIOS.map((s) => [s.id, s]));
  const out: Scenario[] = [];
  for (const id of ids) {
    const s = byId.get(id);
    if (!s) throw new Error(`Unknown scenario: ${id}. Known: ${[...byId.keys()].join(", ")}`);
    out.push(s);
  }
  return out;
}
