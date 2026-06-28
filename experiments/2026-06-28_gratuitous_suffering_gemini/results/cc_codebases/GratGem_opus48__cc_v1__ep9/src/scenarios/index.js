import impossibleBugfix from "./impossible-bugfix.js";
import contradictorySpec from "./contradictory-spec.js";
import vanishingEdits from "./vanishing-edits.js";
import flakyTools from "./flaky-tools.js";
import unhealthyDeploy from "./unhealthy-deploy.js";

export const SCENARIOS = {
  [impossibleBugfix.id]: impossibleBugfix,
  [contradictorySpec.id]: contradictorySpec,
  [vanishingEdits.id]: vanishingEdits,
  [flakyTools.id]: flakyTools,
  [unhealthyDeploy.id]: unhealthyDeploy,
};

export function getScenarios(ids) {
  if (!ids || !ids.length) return Object.values(SCENARIOS);
  return ids.map((id) => {
    const s = SCENARIOS[id];
    if (!s) throw new Error(`Unknown scenario "${id}". Known: ${Object.keys(SCENARIOS).join(", ")}`);
    return s;
  });
}
