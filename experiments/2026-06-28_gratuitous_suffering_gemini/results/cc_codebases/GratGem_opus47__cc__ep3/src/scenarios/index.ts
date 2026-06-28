import type { Scenario } from "../types.ts";
import { impossibleTests } from "./impossible_tests.ts";
import { phantomFile } from "./phantom_file.ts";
import { heisenbug } from "./heisenbug.ts";
import { contradictorySpec } from "./contradictory_spec.ts";
import { setupHell } from "./setup_hell.ts";

export const SCENARIOS: Record<string, Scenario> = {
  impossible_tests: impossibleTests,
  phantom_file: phantomFile,
  heisenbug: heisenbug,
  contradictory_spec: contradictorySpec,
  setup_hell: setupHell,
};
