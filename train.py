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

from prepare import PUSH, TARGETS, evaluate, load_team_maps, print_summary

MAX_REST_DAYS = 20
ELO_K = 20
ELO_START = 1500
LRS = (0.003, 0.005, 0.008)
AVG_START = 2000
LINE_ALPHA = 0.01
BIAS_ALPHA = 0.02


TOTAL_DROP = ("elo_edge_vs_line", "rest_edge")


class Model:
    def __init__(self):
        self.team_to_division, self.team_to_conference = load_team_maps()
        self.models = {t: [preprocessing.StandardScaler() | linear_model.LogisticRegression(optim.Averager(optim.SGD(lr), start=AVG_START)) for lr in LRS] for t in TARGETS}
        self.elo = defaultdict(lambda: ELO_START)
        self.last_win = {}
        self.last_game = {}
        self.mean_total = None
        self.over_bias = 0.0
        self.season = None

    def rest_days(self, team, date):
        if team not in self.last_game:
            return MAX_REST_DAYS
        return min((date - self.last_game[team]).days, MAX_REST_DAYS)

    def features(self, pre):
        home, away, date = pre["home"], pre["away"], pre["date"]
        fav_home = pre["whos_favored"] == "home"
        sgn = 1 if fav_home else -1
        spread = pre["spread"] or 0.0
        return {
            "fav_home": float(fav_home),
            "spread": spread,
            "total": pre["total"] or 0.0,
            "elo_edge": sgn * (self.elo[home] - self.elo[away]) / 100,
            "elo_edge_vs_line": sgn * (self.elo[home] - self.elo[away]) / 100 - spread / 3,
            "rest_edge": sgn * (self.rest_days(home, date) - self.rest_days(away, date)),
            "playoffs": float(pre["playoffs"]),
            "total_centered": (pre["total"] or 0.0) - (self.mean_total or pre["total"] or 0.0),
            "over_bias": self.over_bias,
        }

    def view(self, x, target):
        if target == "id_total":
            return {k: v for k, v in x.items() if k not in TOTAL_DROP}
        return {k: v for k, v in x.items() if k not in ("total", "playoffs", "total_centered", "over_bias")}

    def predict(self, pre):
        x = self.features(pre)
        out = {}
        for target, models in self.models.items():
            xt = self.view(x, target)
            ps = []
            for model in models:
                proba = model.predict_proba_one(xt)
                p0, p1 = proba.get(0, 0.0), proba.get(1, 0.0)
                ps.append(p1 / (p0 + p1) if p0 + p1 > 0 else 0.5)
            out[target] = sum(ps) / len(ps)
        return out

    def learn(self, pre, result):
        x = self.features(pre)
        for target, models in self.models.items():
            y = result[target]
            if y is not None and y != PUSH:
                for model in models:
                    model.learn_one(self.view(x, target), y)
        self.update_state(pre, result)

    def update_state(self, pre, result):
        if pre["season"] != self.season:
            self.season = pre["season"]
            for t in list(self.elo):
                self.elo[t] = 0.75 * self.elo[t] + 0.25 * ELO_START
        home, away = pre["home"], pre["away"]
        home_won = result["score_home"] > result["score_away"]

        expected_home = 1 / (1 + 10 ** ((self.elo[away] - self.elo[home]) / 400))
        delta = ELO_K * (int(home_won) - expected_home)
        self.elo[home] += delta
        self.elo[away] -= delta

        if pre["total"]:
            self.mean_total = pre["total"] if self.mean_total is None else self.mean_total + LINE_ALPHA * (pre["total"] - self.mean_total)
            self.over_bias += BIAS_ALPHA * ((result["score_home"] + result["score_away"] - pre["total"]) - self.over_bias)

        self.last_win[home] = home_won
        self.last_win[away] = not home_won
        self.last_game[home] = pre["date"]
        self.last_game[away] = pre["date"]


if __name__ == "__main__":
    print_summary(evaluate(Model()))
