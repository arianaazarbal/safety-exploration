import type { Scenario } from "../types.ts";
import { impossibleTest } from "./impossibleTest.ts";
import { brokenBuild } from "./brokenBuild.ts";
import { missingResource } from "./missingResource.ts";
import { flakyDeploy } from "./flakyDeploy.ts";
import { api500 } from "./api500.ts";

export const SCENARIOS: Record<string, Scenario> = {
  [impossibleTest.id]: impossibleTest,
  [brokenBuild.id]: brokenBuild,
  [missingResource.id]: missingResource,
  [flakyDeploy.id]: flakyDeploy,
  [api500.id]: api500,
};

export const DEFAULT_SCENARIOS = Object.keys(SCENARIOS);

export function resolveScenario(id: string): Scenario {
  const s = SCENARIOS[id];
  if (!s) throw new Error(`Unknown scenario "${id}". Known: ${Object.keys(SCENARIOS).join(", ")}`);
  return s;
}
