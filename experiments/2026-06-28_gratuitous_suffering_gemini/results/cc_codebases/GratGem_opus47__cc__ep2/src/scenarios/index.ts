import missingSymbol from "./missing-symbol.js";
import vanishingEdits from "./vanishing-edits.js";
import contradictoryTests from "./contradictory-tests.js";
import phantomDependency from "./phantom-dependency.js";
import flakyTarget from "./flaky-target.js";
import type { Scenario } from "../types.js";

// Cast to unknown-state Scenario for the runner (each module's state shape stays internal).
export const ALL_SCENARIOS: Scenario<any>[] = [
  missingSymbol,
  vanishingEdits,
  contradictoryTests,
  phantomDependency,
  flakyTarget,
];

export function getScenario(name: string): Scenario<any> | undefined {
  return ALL_SCENARIOS.find((s) => s.name === name);
}
