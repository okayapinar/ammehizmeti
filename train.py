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

from prepare import PUSH, TARGETS, evaluate, load_team_maps, print_summary

MAX_REST_DAYS = 20
ELO_K = 20.5
ELO_START = 1500
ELO_HOME_ADV = 55
ELO_WARMUP = 75  # league games per season before Elo is updated or used (~5 games per team)
ELO_REGRESS = 0.40  # share of each rating pulled back to ELO_START at a new season
ELO_CATCHUP_GAMES, ELO_CATCHUP_K = 150, 29.75  # faster re-learning (higher K) for this many league games after the warm-up
ELO_EARLY = 275  # league games after the warm-up with a larger home edge and boosted toss-up updates
ELO_EARLY_HCA = 25  # extra home edge in that early window
ELO_TOSSUP_GAP, ELO_TOSSUP_BOOST = 50, 1.16  # early window: games between teams within 50 Elo update 1.16x (separates similar teams)
ELO_WINTER_HCA = -5  # home-edge change in January and February
LRS = (0.003, 0.005, 0.008)
LINE_ALPHA = 0.011
EARLY_THRESHOLDS = (60, 120, 190)
SMOOTH_LABEL_W, SMOOTH_FLIP_W = 0.88, 0.12  # label smoothing: weight of the observed label and of the opposite label
# phase experts: league game after which the late-season expert votes / learns, per target
LATE_PREDICT_FROM = {"id_total": 1000, "id_spread": 1150}
LATE_LEARN_FROM = {"id_total": 950, "id_spread": 1150}
EXPERT_LRS = {"playoff": 0.006, "late": 0.0065}
PLAYOFF_BASE_WEIGHT = 0.45  # playoff games: 0.45 * ensemble + 0.5 * expert (weights sum to 0.95, a mild shrink toward 0)
SPREAD_CAP, TOTAL_CAP = 19.0, 250.0  # line caps for the model features (spread_raw keeps the uncapped spread)
SERIES_CAP = 4  # playoff series game number is capped (games 4+ look alike)
PLAY_IN_SERIES = 0.7  # series_game code for play-in games
VS_LINE_FLOOR = -3.0  # elo_edge_vs_line floor (the upper side never binds)
CONFERENCE = load_team_maps()[1]
SEASON_LENGTH = {2012: 66, 2020: 72, 2021: 72}  # shortened seasons (lockout, pandemic); 82 games otherwise
STAKES_FROM = 60  # team games played before the standings define a playoff-bubble team
TEMPERATURE_LR, TEMPERATURE_MIN = 0.01, 0.05  # online calibration slope on the final logit

# features each target's model sees (the spread model: Elo vs the line; the total model: line level and schedule context)
ELO_FEATURES = ("elo_edge", "elo_edge_vs_line")  # zero during the Elo warm-up
SPREAD_FEATURES = ("fav_home", "spread", *ELO_FEATURES, "rest_edge")
TOTAL_FEATURES = ("fav_home", "spread_raw", "total", "playoffs", "total_centered", "day_game", "wednesday", *(f"early_{thr}" for thr in EARLY_THRESHOLDS),
                  "series_game", "b2b_home", "b2b_away", "b2b_both", "tc_b2b_away", "playoff_saturday")


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def learn_smoothed(model, x, y):
    model.learn_one(x, y, w=SMOOTH_LABEL_W)
    model.learn_one(x, 1 - y, w=SMOOTH_FLIP_W)


class Model:
    """Online logistic ensembles per target (spread: Elo vs the line; total: schedule context vs the line) plus playoff / late-season experts."""

    def __init__(self):
        self.scalers = {t: preprocessing.StandardScaler() for t in TARGETS}
        # ensemble members: (early-season threshold, model) for every lr; the spread view has no early-season indicator,
        # so it needs only one threshold per lr (triplicates would be identical under a plain mean)
        member_thresholds = {"id_total": EARLY_THRESHOLDS, "id_spread": EARLY_THRESHOLDS[:1]}
        self.models = {t: [(thr, linear_model.LogisticRegression(optim.SGD(lr))) for lr in LRS for thr in member_thresholds[t]] for t in TARGETS}
        # temperature calibration of the final probability (a slope on its logit: never changes a pick)
        self.temperature = defaultdict(lambda: 1.0)  # per (target, phase): the final probability is built differently in each phase
        self.pending = {}
        self.elo = defaultdict(lambda: ELO_START)
        self.last_game = {}
        self.mean_total = None
        self.season = None
        self.season_games = 0
        self.series = defaultdict(int)
        self.record = defaultdict(lambda: [0, 0])  # this season's regular-season wins, losses
        self.route_rate = defaultdict(lambda: [1.0, 2.0])  # per rule: routed-side hits, routed games (running hit rate)
        self.routed = {}
        self.playoff_models = {t: linear_model.LogisticRegression(optim.SGD(EXPERT_LRS["playoff"])) for t in TARGETS}  # learn playoff games only
        self.late_models = {t: linear_model.LogisticRegression(optim.SGD(EXPERT_LRS["late"])) for t in TARGETS}  # learn late regular-season games only

    @staticmethod
    def early_view(x, thr):
        # each member sees a single early-season indicator (its own threshold)
        own = f"early_{thr}"
        return {("early_season" if k == own else k): v for k, v in x.items() if k == own or not k.startswith("early_")}

    def expert_view(self, xs):
        # phase experts see the middle early-season indicator
        return self.early_view(xs, EARLY_THRESHOLDS[1])

    def rest_days(self, team, date):
        if team not in self.last_game:
            return MAX_REST_DAYS
        return min((date - self.last_game[team]).days, MAX_REST_DAYS)

    @staticmethod
    def holiday(date):
        # New Year's Day or Martin Luther King Day (third Monday of January)
        if (date.month, date.day) == (1, 1):
            return True
        return date.month == 1 and date.weekday() == 0 and 15 <= date.day <= 21

    def series_game(self, pre):
        # playoff series game number (capped), a fixed code for play-in games, 0 in the regular season
        if pre["playoffs"]:
            return float(min(self.series[frozenset((pre["home"], pre["away"]))] + 1, SERIES_CAP))
        return 0.0 if pre["regular"] else PLAY_IN_SERIES

    def late_rank(self, team):
        # late season (60+ team games, games left): conference rank by wins minus losses, else None
        wins, losses = self.record[team]
        if wins + losses < STAKES_FROM or wins + losses >= SEASON_LENGTH.get(self.season, 82):
            return None
        mates = [(w, l) for t, (w, l) in self.record.items() if t != team and CONFERENCE.get(t) == CONFERENCE.get(team)]
        return 1 + sum(w - l > wins - losses for w, l in mates)

    def bubble(self, team):
        # a playoff-bubble team (seeds 7-10)
        rank = self.late_rank(team)
        return rank is not None and 7 <= rank <= 10

    def lottery(self, team):
        # out of the playoff picture (below seed 8, seed 10 since the play-in)
        rank = self.late_rank(team)
        return rank is not None and rank > (10 if self.season >= 2021 else 8)

    def final_game(self, team):
        # the team's last regular-season game
        return sum(self.record[team]) == SEASON_LENGTH.get(self.season, 82) - 1

    def features(self, pre):
        home, away, date = pre["home"], pre["away"], pre["date"]
        fav_home = pre["whos_favored"] == "home"
        sgn = 1 if fav_home else -1
        spread_raw = pre["spread"] or 0.0  # total model sees the uncapped line
        spread = min(spread_raw, SPREAD_CAP)
        elo_on = float(self.season_games > ELO_WARMUP)  # Elo features are off while Elo is frozen
        elo_edge = sgn * (self.elo[home] - self.elo[away]) / 100  # favorite's Elo edge, in units of 100 Elo
        rest_home, rest_away = self.rest_days(home, date), self.rest_days(away, date)
        b2b_home, b2b_away = float(rest_home <= 1), float(rest_away <= 1)
        total = pre["total"] or 0.0
        line_level = self.mean_total or total  # league line level (the first game is its own level)
        total_centered = 100 * (total - line_level) / (line_level or 1.0)  # percent deviation from the league line level
        return {
            "fav_home": float(fav_home),
            "spread": spread,
            "spread_raw": spread_raw,
            "total": min(total, TOTAL_CAP),
            "elo_edge": elo_edge * elo_on,
            "elo_edge_vs_line": max(VS_LINE_FLOOR, elo_edge - spread / 3) * elo_on,  # Elo edge minus the line (1 point ~ 33 Elo)
            "rest_edge": sgn * (rest_home - rest_away),
            "playoffs": float(pre["playoffs"]),
            "total_centered": total_centered,
            "day_game": float(date.weekday() == 6 or self.holiday(date)),
            "wednesday": float(date.weekday() == 2),
            **{f"early_{thr}": float(self.season_games < thr) for thr in EARLY_THRESHOLDS},
            "series_game": self.series_game(pre),
            "b2b_home": b2b_home,
            "b2b_away": b2b_away,
            "b2b_both": b2b_home * b2b_away,
            "tc_b2b_away": total_centered * b2b_away,
            "playoff_saturday": float(pre["playoffs"] and date.weekday() == 5),
        }

    def view(self, x, target):
        keep = TOTAL_FEATURES if target == "id_total" else SPREAD_FEATURES
        return {k: v for k, v in x.items() if k in keep}

    def phase(self, pre, target, late_from=LATE_PREDICT_FROM):
        if pre["playoffs"]:
            return "playoff"
        if pre["regular"] and self.season_games > late_from[target]:
            return "late"
        return "early" if self.season_games < EARLY_THRESHOLDS[1] else "main"

    def expert(self, pre, target, late_from):
        # phase expert for this game: playoff-only model, late regular-season model, or none
        experts = {"playoff": self.playoff_models, "late": self.late_models}.get(self.phase(pre, target, late_from))
        return experts[target] if experts else None

    def predict(self, pre):
        x = self.features(pre)
        out = {}
        for target, models in self.models.items():
            xs = self.scalers[target].transform_one(self.view(x, target))
            ps = [model.predict_proba_one(self.early_view(xs, g)).get(1, 0.5) for g, model in models]
            out[target] = sum(ps) / len(ps)
            expert = self.expert(pre, target, LATE_PREDICT_FROM)
            if expert is not None:  # blend: 0.5 * expert plus 0.5 (late) or PLAYOFF_BASE_WEIGHT (playoffs) * ensemble
                pe = expert.predict_proba_one(self.expert_view(xs)).get(1, 0.5)
                base_weight = PLAYOFF_BASE_WEIGHT if pre["playoffs"] else 0.5
                out[target] = base_weight * out[target] + 0.5 * pe
        self.route(pre, out)
        return self.calibrate(pre, out)

    def route(self, pre, out):
        # override rules (target -> (rule, side)), each picking its side with the rule's running hit rate:
        # spread - the lone playoff-bubble team covers; total - a team's last regular-season game or two lottery teams go over
        rules = {}
        if pre["regular"] and not pre["playoffs"] and pre["season"] == self.season:  # standings are last season's until the first result
            fav, dog = (pre["home"], pre["away"]) if pre["whos_favored"] == "home" else (pre["away"], pre["home"])
            bubble = int(self.bubble(fav)) - int(self.bubble(dog))
            if bubble:
                rules["id_spread"] = ("bubble", bubble)
            if self.final_game(fav) or self.final_game(dog):
                rules["id_total"] = ("final", 1)
            elif self.lottery(fav) and self.lottery(dog):
                rules["id_total"] = ("lottery", 1)  # two lottery teams: little defence late in the season
        for t, (rule, side) in rules.items():
            self.routed[(self.game_key(pre), t)] = (rule, side)
            hits, n = self.route_rate[rule]
            out[t] = hits / n if side > 0 else 1 - hits / n

    @staticmethod
    def game_key(pre):
        return pre["date"], pre["home"], pre["away"]

    def calibrated(self, z, tk):
        # temperature scaling: a slope on the logit (never changes a pick)
        return 1 / (1 + math.exp(-self.temperature[tk] * z))

    def calibrate(self, pre, out):
        # remember each logit until the result arrives
        key = self.game_key(pre)
        for t in TARGETS:
            z = logit(out[t])
            tk = (t, self.phase(pre, t))
            self.pending[(key, t)] = (z, tk)
            out[t] = self.calibrated(z, tk)
        return out

    def learn_temperature(self, pre, result):
        key = self.game_key(pre)
        for t in TARGETS:
            entry = self.pending.pop((key, t), None)
            if entry is not None and result[t] in (0, 1):
                z, tk = entry
                p = self.calibrated(z, tk)
                self.temperature[tk] = max(TEMPERATURE_MIN, self.temperature[tk] + TEMPERATURE_LR * (result[t] - p) * z)

    def learn(self, pre, result):
        x = self.features(pre)
        self.learn_temperature(pre, result)
        for t in TARGETS:
            rule, side = self.routed.pop((self.game_key(pre), t), (None, 0))
            if side and result[t] in (0, 1):
                self.route_rate[rule][0] += float((result[t] == 1) == (side > 0))
                self.route_rate[rule][1] += 1
        for target, models in self.models.items():
            y = result[target]
            if y is not None and y != PUSH:
                xt = self.view(x, target)
                xs = self.scalers[target].transform_one(xt)  # same scaling as at predict time
                if target == "id_spread" and self.season_games <= ELO_WARMUP:
                    xt = {k: v for k, v in xt.items() if k not in ELO_FEATURES}  # zeroed Elo features do not shape the scaler
                self.scalers[target].learn_one(xt)
                for g, model in models:
                    learn_smoothed(model, self.early_view(xs, g), y)
                expert = self.expert(pre, target, LATE_LEARN_FROM)
                if expert is not None:
                    learn_smoothed(expert, self.expert_view(xs), y)

        self.update_state(pre, result)

    def update_state(self, pre, result):
        if pre["season"] != self.season:
            self.season = pre["season"]
            self.season_games = 0
            self.series = defaultdict(int)
            self.record = defaultdict(lambda: [0, 0])
            for t in list(self.elo):
                self.elo[t] = (1 - ELO_REGRESS) * self.elo[t] + ELO_REGRESS * ELO_START
        self.season_games += 1
        if pre["playoffs"]:
            self.series[frozenset((pre["home"], pre["away"]))] += 1
        elif pre["regular"]:
            home_won = result["score_home"] > result["score_away"]
            self.record[pre["home"]][0 if home_won else 1] += 1
            self.record[pre["away"]][1 if home_won else 0] += 1
        self.update_elo(pre, result)
        if pre["total"]:
            self.mean_total = pre["total"] if self.mean_total is None else self.mean_total + LINE_ALPHA * (pre["total"] - self.mean_total)
        for team in (pre["home"], pre["away"]):
            self.last_game[team] = pre["date"]

    def update_elo(self, pre, result):
        if pre["playoffs"] or self.season_games <= ELO_WARMUP:  # no playoff updates; skip the first ~5 games per team (rotation noise)
            return
        home, away = pre["home"], pre["away"]
        home_won = result["score_home"] > result["score_away"]
        early = self.season_games <= ELO_WARMUP + ELO_EARLY
        if early:
            hca = ELO_HOME_ADV + ELO_EARLY_HCA
        elif pre["date"].month in (1, 2):
            hca = ELO_HOME_ADV + ELO_WINTER_HCA
        else:
            hca = ELO_HOME_ADV
        expected_home = 1 / (1 + 10 ** ((self.elo[away] - self.elo[home] - hca) / 400))
        k = ELO_CATCHUP_K if self.season_games <= ELO_WARMUP + ELO_CATCHUP_GAMES else ELO_K
        delta = k * (int(home_won) - expected_home)
        if early and abs(self.elo[home] - self.elo[away]) < ELO_TOSSUP_GAP:
            delta *= ELO_TOSSUP_BOOST
        self.elo[home] += delta
        self.elo[away] -= delta


if __name__ == "__main__":
    print_summary(evaluate(Model()))
