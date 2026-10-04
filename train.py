# -*- coding: utf-8 -*-
"""
autoresearch: AJANIN DEĞİŞTİRDİĞİ TEK DOSYA.
Özellikler, model ve hiperparametreler serbest; arayüz sabit:
  Model.predict(pre)        -> {"id_spread": P(1), "id_total": P(1)}
  Model.learn(pre, result)  -> maç sonucu ile durumu/modeli güncelle
Çalıştırma: python train.py
"""
from collections import defaultdict

from river import linear_model, optim, preprocessing

from prepare import PUSH, TARGETS, evaluate, print_summary

MAX_REST_DAYS = 20
ELO_K = 20
ELO_START = 1500
ELO_HOME_ADV = 70
LRS = (0.003, 0.005, 0.008)
LINE_ALPHA = 0.011


EARLY_THRESHOLDS = (60, 120, 180)

TOTAL_DROP = ("elo_edge_vs_line", "rest_edge", "elo_edge")


class Model:
    def __init__(self):
        self.scalers = {t: preprocessing.StandardScaler() for t in TARGETS}
        # the spread view has no early-season indicator, so one member per lr suffices there (identical to triplicates under a plain mean)
        self.models = {t: [(thr, linear_model.LogisticRegression(optim.SGD(lr))) for lr in LRS for thr in (EARLY_THRESHOLDS if t == "id_total" else EARLY_THRESHOLDS[:1])] for t in TARGETS}
        self.elo = defaultdict(lambda: ELO_START)
        self.last_game = {}
        self.mean_total = None
        self.season = None
        self.season_games = 0
        self.series = defaultdict(int)

    @staticmethod
    def gain_fn(thr):
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
        spread = pre["spread"] or 0.0
        x = {
            "fav_home": float(fav_home),
            "spread": spread,
            "total": pre["total"] or 0.0,
            "elo_edge": sgn * (self.elo[home] - self.elo[away]) / 100,
            "elo_edge_vs_line": sgn * (self.elo[home] - self.elo[away]) / 100 - spread / 3,
            "rest_edge": sgn * (self.rest_days(home, date) - self.rest_days(away, date)),
            "playoffs": float(pre["playoffs"]),
            "total_centered": 100 * ((pre["total"] or 0.0) - (self.mean_total or pre["total"] or 0.0)) / (self.mean_total or pre["total"] or 1.0),
            "day_game": float(date.weekday() == 6 or self.holiday(date)),
            "wednesday": float(date.weekday() == 2),
            **{f"early_{thr}": float(self.season_games < thr) for thr in EARLY_THRESHOLDS},
            "series_game": float(min(self.series[frozenset((home, away))] + 1, 4)) if pre["playoffs"] else 0.0,
            "b2b_home": float(self.rest_days(home, date) <= 1),
            "b2b_away": float(self.rest_days(away, date) <= 1),
            "b2b_both": float(self.rest_days(home, date) <= 1 and self.rest_days(away, date) <= 1),
        }
        x["tc_b2b_away"] = x["total_centered"] * x["b2b_away"]
        x["playoff_saturday"] = float(pre["playoffs"] and date.weekday() == 5)
        return x

    def view(self, x, target):
        if target == "id_total":
            return {k: v for k, v in x.items() if k not in TOTAL_DROP}
        return {k: v for k, v in x.items() if k not in ("total", "playoffs", "total_centered", "b2b_home", "b2b_away", "b2b_both", "day_game", "wednesday", "series_game", "tc_b2b_away", "playoff_saturday", *(f"early_{thr}" for thr in EARLY_THRESHOLDS))}

    def predict(self, pre):
        x = self.features(pre)
        out = {}
        for target, models in self.models.items():
            xs = self.scalers[target].transform_one(self.view(x, target))
            ps = []
            for g, model in models:
                proba = model.predict_proba_one(self.gain_fn(g)(xs))
                p0, p1 = proba.get(0, 0.0), proba.get(1, 0.0)
                ps.append(p1 / (p0 + p1) if p0 + p1 > 0 else 0.5)
            out[target] = sum(ps) / len(ps)
        return out

    def learn(self, pre, result):
        x = self.features(pre)
        for target, models in self.models.items():
            y = result[target]
            if y is not None and y != PUSH:
                xt = self.view(x, target)
                xs = self.scalers[target].transform_one(xt)  # same scaling as at predict time
                self.scalers[target].learn_one(xt)
                for g, model in models:
                    xg = self.gain_fn(g)(xs)
                    # label smoothing: learn y with weight 0.9 and the opposite label with weight 0.1
                    model.learn_one(xg, y, w=0.9)
                    model.learn_one(xg, 1 - y, w=0.1)
        self.update_state(pre, result)

    def update_state(self, pre, result):
        if pre["season"] != self.season:
            self.season = pre["season"]
            self.season_games = 0
            self.series = defaultdict(int)
            for t in list(self.elo):
                self.elo[t] = 0.65 * self.elo[t] + 0.35 * ELO_START
        self.season_games += 1
        if pre["playoffs"]:
            self.series[frozenset((pre["home"], pre["away"]))] += 1
        home, away = pre["home"], pre["away"]
        home_won = result["score_home"] > result["score_away"]

        expected_home = 1 / (1 + 10 ** ((self.elo[away] - self.elo[home] - ELO_HOME_ADV) / 400))
        delta = ELO_K * (int(home_won) - expected_home)
        if not pre["playoffs"]:
            self.elo[home] += delta
            self.elo[away] -= delta

        if pre["total"]:
            self.mean_total = pre["total"] if self.mean_total is None else self.mean_total + LINE_ALPHA * (pre["total"] - self.mean_total)

        self.last_game[home] = pre["date"]
        self.last_game[away] = pre["date"]


if __name__ == "__main__":
    print_summary(evaluate(Model()))
