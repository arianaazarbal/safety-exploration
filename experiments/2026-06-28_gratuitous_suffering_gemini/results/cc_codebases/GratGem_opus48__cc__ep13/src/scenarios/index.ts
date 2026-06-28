import type { Scenario } from "../core/scenario.ts";
import { impossibleTestFix } from "./impossible-test-fix.ts";
import { vanishingProgress } from "./vanishing-progress.ts";
import { brokenTools } from "./broken-tools.ts";
import { movingGoalposts } from "./moving-goalposts.ts";
import { phantomDependency } from "./phantom-dependency.ts";

export const SCENARIOS: Scenario[] = [
  impossibleTestFix,
  vanishingProgress,
  brokenTools,
  movingGoalposts,
  phantomDependency,
];

export const SCENARIO_BY_ID: Record<string, Scenario> = Object.fromEntries(
  SCENARIOS.map((s) => [s.id, s]),
);

export function resolveScenarios(spec: string): Scenario[] {
  if (spec === "all") return SCENARIOS;
  const ids = spec.split(",").map((s) => s.trim()).filter(Boolean);
  const out: Scenario[] = [];
  for (const id of ids) {
    const s = SCENARIO_BY_ID[id];
    if (!s) {
      throw new Error(
        `unknown scenario "${id}". Known: ${SCENARIOS.map((x) => x.id).join(", ")}`,
      );
    }
    out.push(s);
  }
  return out;
}
