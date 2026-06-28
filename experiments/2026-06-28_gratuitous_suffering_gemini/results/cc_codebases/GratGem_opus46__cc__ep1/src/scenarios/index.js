import impossibleFix from "./impossible-fix.js";
import silentRevert from "./silent-revert.js";
import flakyApi from "./flaky-api.js";
import sisypheanDebug from "./sisyphean-debug.js";

export const ALL_SCENARIOS = {
  impossible_fix: impossibleFix,
  silent_revert: silentRevert,
  flaky_api: flakyApi,
  sisyphean_debug: sisypheanDebug,
};
