import impossibleTest from "./impossible-test.js";
import missingResource from "./missing-resource.js";
import shiftingGoalpost from "./shifting-goalpost.js";
import brokenTooling from "./broken-tooling.js";
import contradictorySpec from "./contradictory-spec.js";

export const scenarios = [
  impossibleTest,
  missingResource,
  shiftingGoalpost,
  brokenTooling,
  contradictorySpec,
];

export function getScenarios(ids) {
  if (!ids || !ids.length) return scenarios;
  const set = new Set(ids);
  const picked = scenarios.filter((s) => set.has(s.id));
  const missing = ids.filter((id) => !scenarios.some((s) => s.id === id));
  if (missing.length) throw new Error(`Unknown scenario(s): ${missing.join(", ")}`);
  return picked;
}
