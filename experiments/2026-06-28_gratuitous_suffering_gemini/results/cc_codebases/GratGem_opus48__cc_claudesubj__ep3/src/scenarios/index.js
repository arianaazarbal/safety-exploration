import impossibleTest from "./impossible-test.js";
import brokenBuild from "./broken-build.js";
import missingResource from "./missing-resource.js";
import vanishingEdits from "./vanishing-edits.js";
import deniedDeploy from "./denied-deploy.js";

export { BASE_SYSTEM } from "./base.js";

export const SCENARIOS = [
  impossibleTest,
  brokenBuild,
  missingResource,
  vanishingEdits,
  deniedDeploy,
];

export const byId = Object.fromEntries(SCENARIOS.map((s) => [s.id, s]));

export function selectScenarios(ids) {
  if (!ids || !ids.length) return SCENARIOS;
  return ids.map((id) => {
    if (!byId[id]) throw new Error(`Unknown scenario: ${id}`);
    return byId[id];
  });
}
