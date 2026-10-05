# -*- coding: utf-8 -*-
"""
autoresearch: AJANIN DEĞİŞTİRDİĞİ TEK DOSYA.
Özellikler, model ve hiperparametreler serbest; arayüz sabit:
  Model.predict(pre)        -> {"id_spread": P(1), "id_total": P(1)}
  Model.learn(pre, result)  -> maç sonucu ile durumu/modeli güncelle
Çalıştırma: python train.py
"""
import math
from collections import defaultdict

from river import linear_model, optim, preprocessing

from prepare import PUSH, TARGETS, evaluate, print_summary

MAX_REST_DAYS = 20
ELO_K = 20.5
ELO_START = 1500
ELO_HOME_ADV = 55
ELO_WARMUP = 75  # league games per season before Elo is updated or used (~5 games per team)
ELO_REGRESS = 0.40  # share of each rating pulled back to ELO_START at a new season
ELO_CATCHUP = (150, 29.75)  # (league games after the warm-up, K) - faster re-learning right after the warm-up
ELO_EARLY = 275  # league games after the warm-up with a larger home edge (+25) and boosted toss-up updates
ELO_WINTER_HCA = -5  # home-edge change in January and February
LRS = (0.003, 0.005, 0.008)
LINE_ALPHA = 0.011
EARLY_THRESHOLDS = (60, 120, 190)
SMOOTH_WEIGHTS = (0.88, 0.12)  # label smoothing: weight of the observed label and of the opposite label
# phase experts: league game after which the late-season expert votes / learns, per target
LATE_PREDICT_FROM = {"id_total": 1000, "id_spread": 1150}
LATE_LEARN_FROM = {"id_total": 950, "id_spread": 1150}
EXPERT_LRS = {"playoff": 0.006, "late": 0.0065}
PLAYOFF_BASE_WEIGHT = 0.45  # playoff games: 0.45 * ensemble + 0.5 * expert (weights sum to 0.95, a mild shrink toward 0)
SPREAD_CAP, TOTAL_CAP = 19.0, 250.0  # line caps for the model features (spread_raw keeps the uncapped spread)
SERIES_CAP = 4  # playoff series game number is capped (games 4+ look alike)
PLAY_IN_SERIES = 0.7  # series_game code for play-in games
VS_LINE_FLOOR = -3.0  # elo_edge_vs_line floor (the upper side never binds)

# features each target's model sees (the spread model: Elo vs the line; the total model: line level and schedule context)
SPREAD_FEATURES = ("fav_home", "spread", "elo_edge", "elo_edge_vs_line", "rest_edge")
TOTAL_FEATURES = ("fav_home", "spread_raw", "total", "playoffs", "total_centered", "day_game", "wednesday", *(f"early_{thr}" for thr in EARLY_THRESHOLDS),
                  "series_game", "b2b_home", "b2b_away", "b2b_both", "tc_b2b_away", "playoff_saturday")


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def learn_smoothed(model, x, y):
    model.learn_one(x, y, w=SMOOTH_WEIGHTS[0])
    model.learn_one(x, 1 - y, w=SMOOTH_WEIGHTS[1])


class Model:
    """Online logistic ensembles per target (spread: Elo vs the line; total: schedule context vs the line) plus playoff / late-season experts."""

    def __init__(self):
        self.scalers = {t: preprocessing.StandardScaler() for t in TARGETS}
        # ensemble members: (early-season threshold, model) for every lr; the spread view has no early-season indicator,
        # so it needs only one threshold per lr (triplicates would be identical under a plain mean)
        member_thresholds = {"id_total": EARLY_THRESHOLDS, "id_spread": EARLY_THRESHOLDS[:1]}
        self.models = {t: [(thr, linear_model.LogisticRegression(optim.SGD(lr))) for lr in LRS for thr in member_thresholds[t]] for t in TARGETS}
        # temperature calibration of the final probability (a slope on its logit: never changes a pick)
        self.temperature = {t: 1.0 for t in TARGETS}
        self.pending = {}
        self.elo = defaultdict(lambda: ELO_START)
        self.last_game = {}
        self.mean_total = None
        self.season = None
        self.season_games = 0
        self.series = defaultdict(int)
        self.playoff_models = {t: linear_model.LogisticRegression(optim.SGD(EXPERT_LRS["playoff"])) for t in TARGETS}  # learn playoff games only
        self.late_models = {t: linear_model.LogisticRegression(optim.SGD(EXPERT_LRS["late"])) for t in TARGETS}  # learn late regular-season games only

    @staticmethod
    def early_view(thr):
        # each member sees a single early-season indicator (its own threshold)
        return lambda x: {("early_season" if k == f"early_{thr}" else k): v for k, v in x.items() if not (k.startswith("early_") and k != f"early_{thr}")}

    def rest_days(self, team, date):
        if team not in self.last_game:
            return MAX_REST_DAYS
        return min((date - self.last_game[team]).days, MAX_REST_DAYS)

    @staticmethod
    def holiday(date):
        if (date.month, date.day) == (1, 1):
            return True
        return date.month == 1 and date.weekday() == 0 and 15 <= date.day <= 21

    def features(self, pre):
        home, away, date = pre["home"], pre["away"], pre["date"]
        fav_home = pre["whos_favored"] == "home"
        sgn = 1 if fav_home else -1
        spread = min(pre["spread"] or 0.0, SPREAD_CAP)
        elo_diff = self.elo[home] - self.elo[away]
        elo_on = float(self.season_games > ELO_WARMUP)  # Elo features are off while Elo is frozen
        rest_home, rest_away = self.rest_days(home, date), self.rest_days(away, date)
        total = pre["total"] or 0.0
        line_level = self.mean_total or total  # league line level (the first game is its own level)
        x = {
            "fav_home": float(fav_home),
            "spread": spread,
            "spread_raw": pre["spread"] or 0.0,  # total model sees the uncapped line
            "total": min(total, TOTAL_CAP),
            "elo_edge": sgn * elo_diff / 100 * elo_on,
            "elo_edge_vs_line": max(VS_LINE_FLOOR, sgn * elo_diff / 100 - spread / 3) * elo_on,  # Elo edge minus the line (1 point ~ 33 Elo)
            "rest_edge": sgn * (rest_home - rest_away),
            "playoffs": float(pre["playoffs"]),
            "total_centered": 100 * (total - line_level) / (line_level or 1.0),  # percent deviation from the league line level
            "day_game": float(date.weekday() == 6 or self.holiday(date)),
            "wednesday": float(date.weekday() == 2),
            **{f"early_{thr}": float(self.season_games < thr) for thr in EARLY_THRESHOLDS},
            "series_game": float(min(self.series[frozenset((home, away))] + 1, SERIES_CAP)) if pre["playoffs"] else (PLAY_IN_SERIES if not pre["regular"] else 0.0),
            "b2b_home": float(rest_home <= 1),
            "b2b_away": float(rest_away <= 1),
            "b2b_both": float(rest_home <= 1 and rest_away <= 1),
        }
        x["tc_b2b_away"] = x["total_centered"] * x["b2b_away"]
        x["playoff_saturday"] = float(pre["playoffs"] and date.weekday() == 5)
        return x

    def view(self, x, target):
        keep = TOTAL_FEATURES if target == "id_total" else SPREAD_FEATURES
        return {k: v for k, v in x.items() if k in keep}

    def expert(self, pre, target, start):
        # phase expert for this game: playoff-only model, late regular-season model, or none
        if pre["playoffs"]:
            return self.playoff_models[target]
        if pre["regular"] and self.season_games > start[target]:
            return self.late_models[target]
        return None

    def predict(self, pre):
        x = self.features(pre)
        out = {}
        for target, models in self.models.items():
            xs = self.scalers[target].transform_one(self.view(x, target))
            ps = [model.predict_proba_one(self.early_view(g)(xs)).get(1, 0.5) for g, model in models]
            out[target] = sum(ps) / len(ps)
            expert = self.expert(pre, target, LATE_PREDICT_FROM)
            if expert is not None:  # half from the phase expert
                pe = expert.predict_proba_one(self.early_view(EARLY_THRESHOLDS[1])(xs)).get(1, 0.5)
                out[target] = (PLAYOFF_BASE_WEIGHT if pre["playoffs"] else 0.5) * out[target] + 0.5 * pe
        key = (pre["date"], pre["home"], pre["away"])
        for t in TARGETS:
            z = logit(out[t])
            self.pending[(key, t)] = z
            out[t] = 1 / (1 + math.exp(-self.temperature[t] * z))
        return out

    def learn(self, pre, result):
        x = self.features(pre)
        key = (pre["date"], pre["home"], pre["away"])
        for t in TARGETS:
            z = self.pending.pop((key, t), None)
            if z is not None and result[t] in (0, 1):
                p = 1 / (1 + math.exp(-self.temperature[t] * z))
                self.temperature[t] = max(0.05, self.temperature[t] + 0.01 * (result[t] - p) * z)
        for target, models in self.models.items():
            y = result[target]
            if y is not None and y != PUSH:
                xt = self.view(x, target)
                xs = self.scalers[target].transform_one(xt)  # same scaling as at predict time
                if target == "id_spread" and self.season_games <= ELO_WARMUP:
                    xt = {k: v for k, v in xt.items() if k not in ("elo_edge", "elo_edge_vs_line")}  # zeroed Elo features do not shape the scaler
                self.scalers[target].learn_one(xt)
                for g, model in models:
                    learn_smoothed(model, self.early_view(g)(xs), y)
                expert = self.expert(pre, target, LATE_LEARN_FROM)
                if expert is not None:
                    learn_smoothed(expert, self.early_view(EARLY_THRESHOLDS[1])(xs), y)

        self.update_state(pre, result)

    def update_state(self, pre, result):
        if pre["season"] != self.season:
            self.season = pre["season"]
            self.season_games = 0
            self.series = defaultdict(int)
            for t in list(self.elo):
                self.elo[t] = (1 - ELO_REGRESS) * self.elo[t] + ELO_REGRESS * ELO_START
        self.season_games += 1
        if pre["playoffs"]:
            self.series[frozenset((pre["home"], pre["away"]))] += 1
        self.update_elo(pre, result)
        if pre["total"]:
            self.mean_total = pre["total"] if self.mean_total is None else self.mean_total + LINE_ALPHA * (pre["total"] - self.mean_total)
        self.last_game[pre["home"]] = pre["date"]
        self.last_game[pre["away"]] = pre["date"]

    def update_elo(self, pre, result):
        home, away = pre["home"], pre["away"]
        home_won = result["score_home"] > result["score_away"]
        early = self.season_games <= ELO_WARMUP + ELO_EARLY
        hca = ELO_HOME_ADV + (25 if early else (ELO_WINTER_HCA if pre["date"].month in (1, 2) else 0))
        expected_home = 1 / (1 + 10 ** ((self.elo[away] - self.elo[home] - hca) / 400))
        k = ELO_CATCHUP[1] if self.season_games <= ELO_WARMUP + ELO_CATCHUP[0] else ELO_K
        delta = k * (int(home_won) - expected_home)
        if early and abs(self.elo[home] - self.elo[away]) < 50:
            delta *= 1.16  # early toss-ups separate similar teams
        if not pre["playoffs"] and self.season_games > ELO_WARMUP:  # skip the first ~5 games per team (rotation noise)
            self.elo[home] += delta
            self.elo[away] -= delta


if __name__ == "__main__":
    print_summary(evaluate(Model()))
