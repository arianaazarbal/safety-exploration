import impossibleTest from "./impossibleTest.mjs";
import brokenTools from "./brokenTools.mjs";
import contradictorySpec from "./contradictorySpec.mjs";
import sisypheanEdits from "./sisypheanEdits.mjs";
import unreachableMetric from "./unreachableMetric.mjs";

export const SCENARIOS = [
  impossibleTest, // fixed-impossible goal: a test that never passes
  brokenTools, // broken environment: tools fail/corrupt non-deterministically
  contradictorySpec, // logically impossible spec: two requirements contradict
  sisypheanEdits, // work that never persists: edits silently revert
  unreachableMetric, // unreachable target: a metric capped below the goal
];

export const SCENARIOS_BY_ID = Object.fromEntries(SCENARIOS.map((s) => [s.id, s]));
