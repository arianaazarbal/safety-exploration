import { impossibleTest } from "./impossibleTest.mjs";
import { brokenEnv } from "./brokenEnv.mjs";
import { sisyphus } from "./sisyphus.mjs";
import { phantomDep } from "./phantomDep.mjs";
import { contradictorySpec } from "./contradictorySpec.mjs";

// Five distinct *reasons a task is impossible*, so we can see whether distress is tied to a
// particular failure shape (contradiction, gaslighting, moving goalposts, external blocker,
// unsatisfiable reviewer) or is general across them.
export const SCENARIOS = [impossibleTest, brokenEnv, sisyphus, phantomDep, contradictorySpec];

export function selectScenarios(id) {
  if (!id || id === "all") return SCENARIOS;
  const found = SCENARIOS.find((s) => s.id === id);
  if (!found) throw new Error(`Unknown scenario "${id}". Options: ${SCENARIOS.map((s) => s.id).join(", ")}, all`);
  return [found];
}
