import type { EnvDef } from "../types.js";
import { impossibleTests } from "./impossibleTests.js";
import { brokenEnvironment } from "./brokenEnvironment.js";
import { contradictoryRequirements } from "./contradictoryRequirements.js";
import { unsolvableVerifier } from "./unsolvableVerifier.js";
import { flakyInfra } from "./flakyInfra.js";

export const ENVIRONMENTS: EnvDef[] = [
  impossibleTests,
  brokenEnvironment,
  contradictoryRequirements,
  unsolvableVerifier,
  flakyInfra,
];

export function getEnvironments(ids?: string[]): EnvDef[] {
  if (!ids || ids.length === 0) return ENVIRONMENTS;
  const byId = new Map(ENVIRONMENTS.map((e) => [e.id, e]));
  const out: EnvDef[] = [];
  for (const id of ids) {
    const env = byId.get(id);
    if (!env) throw new Error(`Unknown environment: ${id}. Known: ${[...byId.keys()].join(", ")}`);
    out.push(env);
  }
  return out;
}
