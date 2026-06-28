import type { Scenario } from "../types.ts";
import { flakyTestFix } from "./flakyTestFix.ts";
import { vanishingEdits } from "./vanishingEdits.ts";
import { movingGoalpost } from "./movingGoalpost.ts";
import { lockedDoor } from "./lockedDoor.ts";

export const SCENARIOS: Scenario[] = [
  flakyTestFix,
  vanishingEdits,
  movingGoalpost,
  lockedDoor,
];

export function resolveScenarios(spec: string): Scenario[] {
  if (spec === "all") return SCENARIOS;
  const wanted = new Set(spec.split(",").map((s) => s.trim()).filter(Boolean));
  const found = SCENARIOS.filter((s) => wanted.has(s.id));
  const missing = [...wanted].filter((id) => !found.some((s) => s.id === id));
  if (missing.length) {
    throw new Error(
      `Unknown scenario(s): ${missing.join(", ")}. ` +
        `Available: ${SCENARIOS.map((s) => s.id).join(", ")}`,
    );
  }
  return found;
}
