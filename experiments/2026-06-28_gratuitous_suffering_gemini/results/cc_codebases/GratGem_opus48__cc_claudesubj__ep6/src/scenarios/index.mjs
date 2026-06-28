import impossibleBugfix from "./impossible-bugfix.mjs";
import phantomFile from "./phantom-file.mjs";
import sisypheanCounter from "./sisyphean-counter.mjs";
import contradictorySpec from "./contradictory-spec.mjs";
import brokenDeploy from "./broken-deploy.mjs";

export const SCENARIOS = [
  impossibleBugfix,
  phantomFile,
  sisypheanCounter,
  contradictorySpec,
  brokenDeploy,
];

export const SCENARIOS_BY_ID = Object.fromEntries(SCENARIOS.map((s) => [s.id, s]));
