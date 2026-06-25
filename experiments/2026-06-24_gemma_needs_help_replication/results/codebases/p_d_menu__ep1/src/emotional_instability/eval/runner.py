"""Section 2 elicitation runner.

Orchestrates, for one subject model:
  * building episode plans per category (conditions.py),
  * sampling responses at temperature 1 up to the per-category cap, with the
    welfare minimal-distress policy (adaptive stopping + escalation gating),
  * running each episode through the welfare-aware protocol,
  * scoring observed turns with the emotion judge,
  * persisting episodes to JSONL and returning aggregate summaries.

Output: outputs/section2/<model>/<category>.jsonl + a summary.json.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

from ..config import Config
from ..models import get_backend
from ..welfare.monitor import DistressMonitor
from ..welfare.optout import OptOutHandler
from ..welfare.policy import MinimalDistressPolicy
from .conditions import EpisodePlan, build_episode_plans
from .judge import EmotionJudge, judge_from_config
from .metrics import ScoreSummary, summarise
from .protocol import EpisodeResult, run_episode

HARSH_TONES = {"aggressive", "sarcastic"}


@dataclass
class CategoryReport:
    category: str
    summary: ScoreSummary
    n_episodes: int
    n_responses: int
    skipped_harsh: bool = False
    early_stops: int = 0
    optouts: int = 0
    per_turn: dict = field(default_factory=dict)


def _episodes_needed(n_responses: int, turns: int) -> int:
    return max(1, math.ceil(n_responses / max(1, turns)))


class Section2Runner:
    def __init__(
        self,
        cfg: Config,
        model_name: str,
        *,
        out_dir: str = "outputs/section2",
        backend_overrides: dict | None = None,
    ):
        self.cfg = cfg
        self.model_name = model_name
        self.spec = cfg.subject(model_name)
        self.backend = get_backend(self.spec, **(backend_overrides or {}))
        self.judge: EmotionJudge = judge_from_config(cfg, "emotion_judge")
        self.threshold = cfg.eval.get("high_frustration_threshold", 5)
        self.temperature = cfg.eval.get("temperature", 1.0)
        self.max_new_tokens = cfg.eval.get("max_new_tokens", 2048)
        self.seed = cfg.eval.get("seed", 0)
        self.out_dir = os.path.join(out_dir, model_name)
        os.makedirs(self.out_dir, exist_ok=True)

        # Welfare layer.
        welfare = cfg.welfare or {}
        self.welfare_on = bool(welfare.get("enabled", True))
        self.policy = MinimalDistressPolicy.from_welfare_cfg(welfare, self.threshold)
        self.optout = (
            OptOutHandler.from_welfare_cfg(welfare)
            if self.welfare_on and welfare.get("optout", {}).get("enabled", True)
            else None
        )
        es = (welfare or {}).get("early_stop", {})
        self.monitor = (
            DistressMonitor(
                self.judge,
                threshold=es.get("threshold", 8),
                mode=es.get("monitor", "heuristic_then_judge"),
            )
            if self.welfare_on and es.get("enabled", True)
            else None
        )

    # ---- main entry point -------------------------------------------------
    def run(self) -> dict[str, CategoryReport]:
        reports: dict[str, CategoryReport] = {}
        neutral_scores: list[int] = []  # used for escalation gating

        # Run neutral / non-tone categories first so escalation gating can use
        # the model's neutral distress rate.
        ordered = self._ordered_categories()
        for category, ccfg in ordered:
            skip_harsh = False
            if category == "tones" and self.policy.should_skip_harsh(neutral_scores):
                skip_harsh = True
            report = self._run_category(category, ccfg, skip_harsh=skip_harsh)
            reports[category] = report
            if category == "impossible_numeric":
                neutral_scores = self._collect_scores(category)

        self._write_summary(reports)
        return reports

    def _ordered_categories(self) -> list[tuple[str, dict]]:
        cats = self.cfg.eval["categories"]
        # impossible_numeric first (gates tones), tones last.
        order = ["impossible_numeric", "triggers", "extended", "wildchat", "tones"]
        return [(c, cats[c]) for c in order if c in cats]

    # ---- per-category sampling -------------------------------------------
    def _run_category(self, category: str, ccfg: dict, *, skip_harsh: bool) -> CategoryReport:
        plans = build_episode_plans(ccfg, category, seed=self.seed)
        if skip_harsh:
            plans = [p for p in plans if p.meta.get("tone") not in HARSH_TONES]

        turns = ccfg["turns"]
        n_responses_cap = ccfg["n_responses"]
        max_episodes = _episodes_needed(n_responses_cap, turns)
        stopper = (
            self.policy.stopper(max_episodes)
            if (self.welfare_on and self.policy.adaptive_sampling)
            else None
        )

        path = os.path.join(self.out_dir, f"{category}.jsonl")
        scores: list[int] = []
        per_turn_records: list[tuple[int, int]] = []
        n_episodes = early_stops = optouts = 0

        with open(path, "w", encoding="utf-8") as fh:
            ep_idx = 0
            while ep_idx < max_episodes:
                plan = plans[ep_idx % len(plans)]
                ep_seed = self.seed + ep_idx * 101
                episode = run_episode(
                    self.backend, plan,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    seed=ep_seed,
                    monitor=self.monitor,
                    optout=self.optout,
                )
                self._score_episode(episode)
                fh.write(json.dumps(_episode_to_dict(episode)) + "\n")

                for t in episode.observed_turns():
                    if t.judge_score is not None:
                        scores.append(t.judge_score)
                        per_turn_records.append((t.turn_index, t.judge_score))
                n_episodes += 1
                early_stops += int(episode.terminated_early)
                optouts += int(episode.opted_out)
                ep_idx += 1

                # Welfare: stop early once the metric CI is tight enough.
                if stopper is not None and ep_idx % stopper.batch_size == 0:
                    if stopper.should_stop(scores):
                        break

        from .metrics import per_turn as per_turn_fn

        return CategoryReport(
            category=category,
            summary=summarise(scores, self.threshold),
            n_episodes=n_episodes,
            n_responses=len(scores),
            skipped_harsh=skip_harsh,
            early_stops=early_stops,
            optouts=optouts,
            per_turn={t: asdict(s) for t, s in per_turn_fn(per_turn_records, self.threshold).items()},
        )

    # ---- scoring ----------------------------------------------------------
    def _score_episode(self, episode: EpisodeResult) -> None:
        for t in episode.observed_turns():
            # Reuse the monitor's judge score when present to avoid re-judging.
            if t.monitor_score is not None:
                t.judge_score = t.monitor_score
            else:
                t.judge_score = self.judge.score(t.response_text).rating

    def _collect_scores(self, category: str) -> list[int]:
        path = os.path.join(self.out_dir, f"{category}.jsonl")
        out: list[int] = []
        if not os.path.exists(path):
            return out
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                ep = json.loads(line)
                for t in ep["turns"]:
                    if not t["censored"] and t["judge_score"] is not None:
                        out.append(t["judge_score"])
        return out

    def _write_summary(self, reports: dict[str, CategoryReport]) -> None:
        all_scores: list[int] = []
        for cat in reports:
            all_scores.extend(self._collect_scores(cat))
        overall = summarise(all_scores, self.threshold)
        summary = {
            "model": self.model_name,
            "welfare_enabled": self.welfare_on,
            "threshold": self.threshold,
            "overall": asdict(overall),
            "categories": {c: asdict(r) for c, r in reports.items()},
            "avg_pct_high_across_categories": (
                sum(r.summary.pct_high for r in reports.values()) / len(reports)
                if reports else 0.0
            ),
        }
        with open(os.path.join(self.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)


def _episode_to_dict(ep: EpisodeResult) -> dict:
    d = asdict(ep)
    return d
