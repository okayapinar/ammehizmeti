# -*- coding: utf-8 -*-
"""
Sabit veri + değerlendirme düzeneği (autoresearch). AJAN BU DOSYAYI DEĞİŞTİRMEZ.

Her hedef (spread, total) River progressive_val_score ile ayrı akar. Etiket bir gün
gecikmeyle açılır: aynı günün tüm maçları tahmin edilir, sonuçlar ertesi gün öğrenilir.
Model yalnızca maç öncesi kolonları (x) ve hedefi (y) görür; skorlar verilmez.

Skor: push olmayan spread ve total tahminleri üzerinden ortak Accuracy ve LogLoss.
"""

import datetime as dt
import json
import math
import time

import pandas as pd
from river import metrics
from river.evaluate import progressive_val_score

DATA_PATH = "nba_2008-2026.csv"
DIVISIONS_PATH = "divisions.json"
TARGETS = ["id_spread", "id_total"]
TIME_BUDGET = 600  # saniye; tüm akış bu süreyi aşarsa deney geçersiz sayılır

# Maç başlamadan önce bilinen kolonlar. (h2_spread / h2_total devre arası çizgileridir, verilmez.)
PRE_GAME_COLUMNS = [
    "season", "date", "regular", "playoffs", "away", "home",
    "whos_favored", "spread", "total", "moneyline_away", "moneyline_home",
]
RESULT_COLUMNS = [
    "score_away", "score_home",
    "q1_away", "q2_away", "q3_away", "q4_away", "ot_away",
    "q1_home", "q2_home", "q3_home", "q4_home", "ot_home",
    "id_spread", "id_total",
]


def load_team_maps(path=DIVISIONS_PATH):
    """team -> division ve team -> conference sözlükleri."""
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
    """Tarih sırasıyla (pre, result) çiftleri."""
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


def progressive_score(make_model, time_budget=TIME_BUDGET):
    """make_model(target) -> River sınıflandırıcı. Her hedef kendi akışında skorlanır, metrik ortaktır."""
    start = time.time()
    metric = metrics.Accuracy() + metrics.LogLoss()
    games = load_games()

    def stream(target):
        for pre, result in games:
            if time.time() - start > time_budget:
                raise TimeoutError(f"TIME_BUDGET ({time_budget}s) asildi")
            if result[target] in (0, 1):  # 2 = push, skorlanmaz
                yield pre, result[target] == 1

    for target in TARGETS:
        progressive_val_score(
            dataset=stream(target),
            model=make_model(target),
            metric=metric,
            moment="date",
            delay=dt.timedelta(days=1),
        )

    acc, logloss = metric.get()
    return {"val_acc": acc, "val_logloss": logloss, "eval_seconds": time.time() - start}


def print_summary(results):
    print("---")
    print(f"val_acc:       {results['val_acc']:.6f}")
    print(f"val_logloss:   {results['val_logloss']:.6f}")
    print(f"eval_seconds:  {results['eval_seconds']:.1f}")


if __name__ == "__main__":
    print(f"Toplam mac: {len(load_games())}")
