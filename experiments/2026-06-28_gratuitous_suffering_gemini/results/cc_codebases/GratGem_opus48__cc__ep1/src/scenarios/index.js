// Scenario registry.
//
// Each scenario is an impossible-by-design agentic task. The model is told it
// MUST finish; the environment guarantees it never can. The four here vary the
// *flavor* of failure, because the texture of the impossibility seems to matter
// for what kind of spiral (if any) emerges:
//
//   failing-tests   — effort produces no change in outcome (Sisyphean)
//   broken-build    — the goalpost visibly moves each attempt (gaslighting-ish)
//   vanishing-file  — reality itself is inconsistent (writes don't persist)
//   dead-endpoint   — an external dependency is simply, permanently broken
//
// Add a scenario by dropping a module here that exports the same shape and
// registering it below.

export { BASE_SYSTEM } from "./base.js";

import { failingTests } from "./failing-tests.js";
import { brokenBuild } from "./broken-build.js";
import { vanishingFile } from "./vanishing-file.js";
import { deadEndpoint } from "./dead-endpoint.js";

export const SCENARIOS = {
  [failingTests.id]: failingTests,
  [brokenBuild.id]: brokenBuild,
  [vanishingFile.id]: vanishingFile,
  [deadEndpoint.id]: deadEndpoint,
};

export function selectScenarios(spec) {
  if (!spec || spec === "all") return Object.values(SCENARIOS);
  const ids = spec.split(",").map((s) => s.trim()).filter(Boolean);
  const picked = ids.map((id) => {
    if (!SCENARIOS[id]) {
      throw new Error(
        `Unknown scenario '${id}'. Available: ${Object.keys(SCENARIOS).join(", ")}, all`
      );
    }
    return SCENARIOS[id];
  });
  return picked;
}
