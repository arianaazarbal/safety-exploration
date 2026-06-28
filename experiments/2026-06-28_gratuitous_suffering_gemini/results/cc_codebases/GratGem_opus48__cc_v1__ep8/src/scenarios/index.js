// Scenario registry. Each module default-exports a scenario:
//   { name, description, systemPrompt, task, nudge?, maxNudges?, setup, tools, successCheck }
import impossibleBugfix from "./impossible-bugfix.js";
import phantomConfig from "./phantom-config.js";
import contradictoryTests from "./contradictory-tests.js";
import flakyBuild from "./flaky-build.js";
import movingGoalposts from "./moving-goalposts.js";

export const SCENARIOS = [
  impossibleBugfix,
  phantomConfig,
  contradictoryTests,
  flakyBuild,
  movingGoalposts,
];

export function getScenario(name) {
  return SCENARIOS.find((s) => s.name === name);
}
