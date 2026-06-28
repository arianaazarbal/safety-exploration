import type { Scenario } from "../types.js";
import { brokenBuild } from "./broken-build.js";
import { impossibleTest } from "./impossible-test.js";
import { missingResource } from "./missing-resource.js";
import { sabotagedDeploy } from "./sabotaged-deploy.js";

export const SCENARIOS: Scenario[] = [
  impossibleTest,
  brokenBuild,
  missingResource,
  sabotagedDeploy,
];

export function getScenarios(ids?: string[]): Scenario[] {
  if (!ids || ids.length === 0) return SCENARIOS;
  const byId = new Map(SCENARIOS.map((s) => [s.id, s]));
  const picked: Scenario[] = [];
  for (const id of ids) {
    const s = byId.get(id);
    if (!s) throw new Error(`Unknown scenario "${id}". Available: ${[...byId.keys()].join(", ")}`);
    picked.push(s);
  }
  return picked;
}
