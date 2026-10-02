# -*- coding: utf-8 -*-
"""
Sabit veri + değerlendirme düzeneği (autoresearch). AJAN BU DOSYAYI DEĞİŞTİRMEZ.

Maçlar tarih sırasıyla, gün gün akıtılır (prequential / online):
  1. O günün tüm maçları için model.predict(pre_game) çağrılır (sadece maç öncesi bilgi).
  2. Ardından aynı maçlar için model.learn(pre_game, result) çağrılır (skorlar + hedefler).
Böylece model hiçbir maçın sonucunu tahminden önce göremez (sızıntı yapısal olarak engellenir).

Skor: VAL_START_SEASON ve sonrası sezonlarda, push (2) olmayan maçlar üzerinden
id_spread ve id_total doğruluklarının ortalaması -> val_acc (yüksek = iyi).
"""
import json
import math
import time
from itertools import groupby

import pandas as pd

DATA_PATH = "nba_2008-2026.csv"
DIVISIONS_PATH = "divisions.json"
TARGETS = ["id_spread", "id_total"]
PUSH = 2
VAL_START_SEASON = 2024
TIME_BUDGET = 300  # saniye; tüm akış bu süreyi aşarsa deney geçersiz sayılır
EPS = 1e-6

# Maç başlamadan önce bilinen kolonlar. (h2_spread / h2_total devre arası çizgileridir, verilmez.)
PRE_GAME_COLUMNS = [
    "season",
    "date",
    "regular",
    "playoffs",
    "away",
    "home",
    "whos_favored",
    "spread",
    "total",
    "moneyline_away",
    "moneyline_home",
]
RESULT_COLUMNS = [
    "score_away",
    "score_home",
    "q1_away",
    "q2_away",
    "q3_away",
    "q4_away",
    "ot_away",
    "q1_home",
    "q2_home",
    "q3_home",
    "q4_home",
    "ot_home",
    "id_spread",
    "id_total",
]


def load_team_maps(path=DIVISIONS_PATH):
    with open(path) as f:
        divisions = json.load(f)

    team_to_division = {}
    team_to_conference = {}
    for conference, divs in divisions.items():
        for division, teams in divs.items():
            for team in teams:
                team_to_division[team] = division
                team_to_conference[team] = conference
    return team_to_division, team_to_conference


def _clean(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy skaler -> python
        return value.item()
    return value


def load_games(path=DATA_PATH):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date", kind="stable")

    games = []
    for row in df.to_dict("records"):
        pre = {col: _clean(row[col]) for col in PRE_GAME_COLUMNS}
        result = {col: _clean(row[col]) for col in RESULT_COLUMNS}
        for target in TARGETS:
            if result[target] is not None:
                result[target] = int(result[target])
        games.append((pre, result))
    return games


def evaluate(model, time_budget=TIME_BUDGET):
    """model.predict(pre) -> {"id_spread": P(1), "id_total": P(1)}; model.learn(pre, result)."""
    games = load_games()
    stats = {t: {"n": 0, "correct": 0, "logloss": 0.0} for t in TARGETS}
    start = time.time()

    for _, day in groupby(games, key=lambda g: g[0]["date"]):
        day = list(day)
        preds = [model.predict(dict(pre)) for pre, _ in day]

        for (pre, result), pred in zip(day, preds):
            if pre["season"] < VAL_START_SEASON:
                continue
            for target in TARGETS:
                y = result[target]
                if y is None or y == PUSH:
                    continue
                p = min(max(float(pred[target]), EPS), 1 - EPS)
                s = stats[target]
                s["n"] += 1
                s["correct"] += int((p >= 0.5) == (y == 1))
                s["logloss"] -= math.log(p if y == 1 else 1 - p)

        for pre, result in day:
            model.learn(dict(pre), dict(result))

        if time.time() - start > time_budget:
            raise TimeoutError(f"TIME_BUDGET ({time_budget}s) asildi")

    acc = {t: stats[t]["correct"] / stats[t]["n"] for t in TARGETS}
    logloss = {t: stats[t]["logloss"] / stats[t]["n"] for t in TARGETS}
    return {
        "val_acc": sum(acc.values()) / len(TARGETS),
        "val_acc_spread": acc["id_spread"],
        "val_acc_total": acc["id_total"],
        "val_logloss": sum(logloss.values()) / len(TARGETS),
        "num_val_games": stats["id_spread"]["n"],
        "eval_seconds": time.time() - start,
    }


def print_summary(results):
    print("---")
    print(f"val_acc:          {results['val_acc']:.6f}")
    print(f"val_acc_spread:   {results['val_acc_spread']:.6f}")
    print(f"val_acc_total:    {results['val_acc_total']:.6f}")
    print(f"val_logloss:      {results['val_logloss']:.6f}")
    print(f"num_val_games:    {results['num_val_games']}")
    print(f"eval_seconds:     {results['eval_seconds']:.1f}")


if __name__ == "__main__":
    games = load_games()
    n_val = sum(pre["season"] >= VAL_START_SEASON for pre, _ in games)
    print(f"Toplam mac: {len(games)}, val mac (sezon >= {VAL_START_SEASON}): {n_val}")
