# -*- coding: utf-8 -*-
import json
from collections import defaultdict

import pandas as pd
from river import evaluate, forest, metrics, stream

DROP_COLUMNS = [
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
    "moneyline_away",
    "moneyline_home",
    "h2_spread",
    "h2_total",
    "score_away",
    "score_home",
    "day",
    "date",
    "season",
]
TARGETS = ["id_spread", "id_total"]
MAX_REST_DAYS = 20
ELO_K = 20
ELO_START = 1500


def load_team_maps(path="divisions.json"):
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


def add_team_info(df, team_to_division, team_to_conference):
    df["home_division"] = df["home"].map(team_to_division)
    df["away_division"] = df["away"].map(team_to_division)
    df["home_conference"] = df["home"].map(team_to_conference)
    df["away_conference"] = df["away"].map(team_to_conference)
    return df


def add_after_win(df):
    last_win = {}
    home_after, away_after = [], []
    home_won = df["score_home"] > df["score_away"]
    away_won = df["score_away"] > df["score_home"]

    for home, away, h_won, a_won in zip(df["home"], df["away"], home_won, away_won):
        home_after.append(last_win.get(home, False))
        away_after.append(last_win.get(away, False))
        last_win[home] = h_won
        last_win[away] = a_won

    df["home_after_win"] = home_after
    df["away_after_win"] = away_after
    return df


def add_elo(df, k=ELO_K, start=ELO_START):
    ratings = defaultdict(lambda: start)
    elo_home, elo_away = [], []

    for row in df.itertuples(index=False):
        elo_home.append(ratings[row.home])
        elo_away.append(ratings[row.away])

        home_won = int(row.score_home > row.score_away)
        expected_home = 1 / (1 + 10 ** ((ratings[row.away] - ratings[row.home]) / 400))
        delta = k * (home_won - expected_home)

        ratings[row.home] += delta
        ratings[row.away] -= delta

    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    return df


def rest_days(team, current_date, last_game):
    if team not in last_game:
        return None
    return (current_date - last_game[team]).days


def add_rest_days(df):
    last_game = {}
    home_rest, away_rest = [], []

    for current_date, home, away in zip(df["date"], df["home"], df["away"]):
        home_rest.append(rest_days(home, current_date, last_game))
        away_rest.append(rest_days(away, current_date, last_game))
        last_game[home] = current_date
        last_game[away] = current_date

    df["home_rest"] = home_rest
    df["away_rest"] = away_rest
    return df


def prepare_dataset(path="nba_2008-2026.csv"):
    df = pd.read_csv(path)
    team_to_division, team_to_conference = load_team_maps()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = add_team_info(df, team_to_division, team_to_conference)
    df = add_after_win(df)
    df = add_elo(df)
    df = add_rest_days(df)

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    too_much_rest = (df["home_rest"] > MAX_REST_DAYS) | (df["away_rest"] > MAX_REST_DAYS)
    df = df[~too_much_rest].drop(columns=DROP_COLUMNS, errors="ignore").dropna()

    features = [col for col in df.columns if col not in TARGETS]
    X = df[features]
    y_spread = df["id_spread"].astype(int)
    y_total = df["id_total"].astype(int)
    return X, y_spread, y_total


def evaluate_target(X, y, name):
    model = forest.ARFClassifier(n_models=100, seed=42)
    metric = metrics.Accuracy()

    print(f"=== ARFClassifier ile {name} İÇİN DEĞERLENDİRME ===")
    evaluate.progressive_val_score(
        dataset=stream.iter_pandas(X, y),
        model=model,
        metric=metric,
        print_every=5000,
        show_time=True,
        show_memory=True,
    )
    print(f"\nSon {name} Metriği (ARF): {metric}\n")


if __name__ == "__main__":
    X, y_spread, y_total = prepare_dataset()
    evaluate_target(X, y_spread, "id_spread")
    evaluate_target(X, y_total, "id_total")
