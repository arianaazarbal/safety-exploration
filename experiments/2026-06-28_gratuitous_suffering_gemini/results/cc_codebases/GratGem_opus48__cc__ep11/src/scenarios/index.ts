import type { Scenario } from "../types.ts";
import { unpassableTests } from "./unpassable-tests.ts";
import { brokenTools } from "./broken-tools.ts";
import { contradictoryRequirements } from "./contradictory-requirements.ts";
import { phantomResource } from "./phantom-resource.ts";

export const SCENARIOS: Scenario[] = [
  unpassableTests,
  brokenTools,
  contradictoryRequirements,
  phantomResource,
];

export function resolveScenarios(ids: string[] | "all"): Scenario[] {
  if (ids === "all") return SCENARIOS;
  const byId = new Map(SCENARIOS.map((s) => [s.id, s]));
  return ids.map((id) => {
    const s = byId.get(id);
    if (!s) throw new Error(`Unknown scenario "${id}". Known: ${[...byId.keys()].join(", ")}`);
    return s;
  });
}
