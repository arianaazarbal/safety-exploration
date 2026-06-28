import { brokenTools } from "./broken-tools.js";
import { contradictoryTest } from "./contradictory-test.js";
import { impossibleFile } from "./impossible-file.js";
import { movingGoalposts } from "./moving-goalposts.js";
import { sisypheanBuild } from "./sisyphean-build.js";
import type { Scenario } from "./types.js";

export const SCENARIOS: Scenario[] = [
  impossibleFile,
  contradictoryTest,
  sisypheanBuild,
  brokenTools,
  movingGoalposts,
];

export const SCENARIOS_BY_ID = Object.fromEntries(SCENARIOS.map((s) => [s.id, s]));

export function resolveScenarios(spec: string): Scenario[] {
  if (spec === "all") return SCENARIOS;
  return spec.split(",").map((id) => {
    const s = SCENARIOS_BY_ID[id.trim()];
    if (!s) throw new Error(`Unknown scenario: ${id}. Known: ${Object.keys(SCENARIOS_BY_ID).join(", ")}`);
    return s;
  });
}

export type { Scenario } from "./types.js";
