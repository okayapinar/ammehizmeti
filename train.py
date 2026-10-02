# -*- coding: utf-8 -*-
"""
autoresearch: AJANIN DEĞİŞTİRDİĞİ TEK DOSYA.
Özellikler, model ve hiperparametreler serbest; arayüz sabit:
  Model.predict(pre)        -> {"id_spread": P(1), "id_total": P(1)}
  Model.learn(pre, result)  -> maç sonucu ile durumu/modeli güncelle
Çalıştırma: python train.py
"""
from collections import defaultdict

from river import forest

from prepare import PUSH, TARGETS, evaluate, load_team_maps, print_summary

MAX_REST_DAYS = 20
ELO_K = 20
ELO_START = 1500
N_MODELS = 10
SEED = 42


class Model:
    def __init__(self):
        self.team_to_division, self.team_to_conference = load_team_maps()
        self.models = {t: forest.ARFClassifier(n_models=N_MODELS, seed=SEED) for t in TARGETS}
        self.elo = defaultdict(lambda: ELO_START)
        self.last_win = {}
        self.last_game = {}

    def rest_days(self, team, date):
        if team not in self.last_game:
            return MAX_REST_DAYS
        return min((date - self.last_game[team]).days, MAX_REST_DAYS)

    def features(self, pre):
        home, away, date = pre["home"], pre["away"], pre["date"]
        return {
            "regular": pre["regular"],
            "playoffs": pre["playoffs"],
            "home": home,
            "away": away,
            "whos_favored": pre["whos_favored"],
            "spread": pre["spread"] or 0.0,
            "total": pre["total"],
            "home_division": self.team_to_division[home],
            "away_division": self.team_to_division[away],
            "home_conference": self.team_to_conference[home],
            "away_conference": self.team_to_conference[away],
            "home_after_win": self.last_win.get(home, False),
            "away_after_win": self.last_win.get(away, False),
            "elo_home": self.elo[home],
            "elo_away": self.elo[away],
            "home_rest": self.rest_days(home, date),
            "away_rest": self.rest_days(away, date),
            "month": date.month,
            "day_of_week": date.weekday(),
        }

    def predict(self, pre):
        x = self.features(pre)
        out = {}
        for target, model in self.models.items():
            proba = model.predict_proba_one(x)
            p0, p1 = proba.get(0, 0.0), proba.get(1, 0.0)
            out[target] = p1 / (p0 + p1) if p0 + p1 > 0 else 0.5
        return out

    def learn(self, pre, result):
        x = self.features(pre)
        for target, model in self.models.items():
            y = result[target]
            if y is not None and y != PUSH:
                model.learn_one(x, y)
        self.update_state(pre, result)

    def update_state(self, pre, result):
        home, away = pre["home"], pre["away"]
        home_won = result["score_home"] > result["score_away"]

        expected_home = 1 / (1 + 10 ** ((self.elo[away] - self.elo[home]) / 400))
        delta = ELO_K * (int(home_won) - expected_home)
        self.elo[home] += delta
        self.elo[away] -= delta

        self.last_win[home] = home_won
        self.last_win[away] = not home_won
        self.last_game[home] = pre["date"]
        self.last_game[away] = pre["date"]


if __name__ == "__main__":
    print_summary(evaluate(Model()))
