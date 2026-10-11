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

import numpy as np
from river import linear_model, optim, preprocessing
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from prepare import TARGETS, evaluate, load_team_maps, print_summary

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
LRS = (0.003, 0.005, 0.008)
LINE_ALPHA = 0.011
EARLY_THRESHOLDS = (60, 120, 190)
SMOOTH_LABEL_W = 0.88  # label smoothing: weight of the observed label (the opposite label gets the rest)
LATE_PREDICT_LEFT, LATE_LEARN_LEFT = 230, 280  # late-season total expert: votes / learns in the last this many league games of the regular season
LATE_EXPERT_LR = 0.0065
SPREAD_CAP, TOTAL_CAP = 19.0, 250.0  # line caps for the model features (spread_raw keeps the uncapped spread)
SERIES_CAP = 4  # playoff series game number is capped (games 4+ look alike)
PLAY_IN_SERIES = 0.7  # series_game code for the other non-regular games (play-in, NBA Cup final)
VS_LINE_FLOOR = -3.0  # elo_edge_vs_line floor (the upper side never binds)
CONFERENCE = load_team_maps()[1]
SEASON_LENGTH = {2012: 66, 2021: 72}  # shortened seasons (lockout, pandemic); 82 otherwise (2019-20 ended unevenly at 64-75 games: no final game)
STAKES_FROM = 60  # team games played before the standings define a playoff-bubble or lottery team
TEMPERATURE_LR, TEMPERATURE_MIN = 0.01, 0.05  # online calibration slope on the final logit
RIDGE_ALPHA = 10.0  # penalty of the line-error ridges (playoffs, regular-season segments)
RIDGE_MIN = 60  # past games a line-error ridge needs
ALTITUDE = ("den", "utah")  # high-altitude home courts
# regular-season segments with their own line-error ridge: name -> (target, weight of the ridge probability)
SEGMENTS = {"altitude_b2b": ("id_spread", 1.0), "rest_mismatch": ("id_total", 0.5), "ats_streak": ("id_spread", 0.5),
            "prev_ot": ("id_spread", 0.5), "contender_b2b": ("id_spread", 0.5),
            "co_week": ("id_spread", 0.5)}
ATS_STREAK_GAMES, ATS_STREAK_MEAN = 3, 14.0  # a team's last 3 line errors share a sign with mean |error| >= 14 points
COMMON_DAYS = 7  # co_week: both teams met the same opponent within the last 7 days
CONTENDER_GAMES, CONTENDER_PCT = 41, 0.6  # load management: a .600+ team with 41+ games played
FAV_LEAN, FAV_LEAN_ATS_GAP = 0.06051, -2.75  # fav_lean router: model spread logit above this, favourite's last-5 ATS form minus the underdog's above this

# features each target's model sees (the spread model: Elo vs the line; the total model: line level and schedule context)
ELO_FEATURES = ("elo_edge", "elo_edge_vs_line")  # raw 0 during the Elo warm-up (after scaling a constant: an implicit warm-up indicator)
SPREAD_FEATURES = ("fav_home", "spread", *ELO_FEATURES, "rest_edge")
TOTAL_FEATURES = ("fav_home", "spread_raw", "total", "playoffs", "total_centered", "day_game", "wednesday", *(f"early_{thr}" for thr in EARLY_THRESHOLDS),
                  "series_game", "b2b_home", "b2b_away", "b2b_both", "tc_b2b_away", "playoff_saturday")


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_line_ridge(rows):
    # ridge of the line error on (context, error) rows: (scaler, ridge, residual sd), or None with too few rows
    if len(rows) < RIDGE_MIN:
        return None
    X, y = np.array([f for f, _ in rows]), np.array([y for _, y in rows])
    scaler = StandardScaler().fit(X)
    reg = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(X), y)
    return scaler, reg, (y - reg.predict(scaler.transform(X))).std()


def line_ridge_prob(fit, f):
    # P(line error > 0) = Phi(predicted error / residual sd)
    scaler, reg, sd = fit
    return 0.5 * (1 + math.erf(reg.predict(scaler.transform([f]))[0] / sd / math.sqrt(2)))


def learn_smoothed(model, x, y):
    model.learn_one(x, y, w=SMOOTH_LABEL_W)
    model.learn_one(x, 1 - y, w=1 - SMOOTH_LABEL_W)


class Model:
    """Online logistic ensembles per target (spread: Elo vs the line; total: schedule context vs the line) plus a late-season total expert, routers and line-error ridges (playoffs, regular-season segments)."""

    def __init__(self):
        self.scalers = {t: preprocessing.StandardScaler() for t in TARGETS}
        # ensemble members: (early-season threshold, model) for every lr; the spread view has no early-season indicator (None)
        member_thresholds = {"id_total": EARLY_THRESHOLDS, "id_spread": (None,)}
        self.models = {t: [(thr, linear_model.LogisticRegression(optim.SGD(lr))) for lr in LRS for thr in member_thresholds[t]] for t in TARGETS}
        # temperature calibration of the final probability (a slope on its logit: never changes a pick)
        self.temperature = defaultdict(lambda: 1.0)  # per (target, phase): the final probability is built differently in each phase
        self.pending = {}  # game key -> what predict() leaves for learn(): {"logit", "routed", "playoff", "segment"} by target / segment
        self.elo = defaultdict(lambda: ELO_START)
        self.last_game = {}
        self.mean_total = None
        self.season = None
        self.season_games = 0
        self.series = defaultdict(int)
        # playoff series -> its previous game's ({team: margin minus line-implied margin}, total residual, overtime points, total line)
        self.series_last = {}
        self.playoff_opponents = defaultdict(set)  # this postseason's opponents per team (round number)
        self.playoff_rows = {t: [] for t in TARGETS}  # past playoff games: (season, context, line error: favourite's margin / total residual)
        self.playoff_reg = {}
        self.segment_rows = {seg: [] for seg in SEGMENTS}  # past segment games: (season, context, line error)
        self.segment_reg = {}
        self.ridge_season = None  # the season the line-error ridges were last fit for
        self.line_errors = defaultdict(list)  # (season, team) -> its regular-season line errors (margin minus line-implied margin)
        self.recent = defaultdict(list)  # team -> its regular-season games: (date, opponent, its line error)
        self.last_ot = {}  # (season, team) -> whether its last regular-season game went to overtime
        self.record = defaultdict(lambda: [0, 0])  # this season's regular-season wins, losses
        self.route_rate = defaultdict(lambda: [1.0, 2.0])  # per rule: routed-side hits, routed games (running hit rate)
        self.late_model = linear_model.LogisticRegression(optim.SGD(LATE_EXPERT_LR))  # the total's late regular-season expert

    @staticmethod
    def early_view(x, thr):
        # each member sees a single early-season indicator (its own threshold)
        own = f"early_{thr}"
        return {("early_season" if k == own else k): v for k, v in x.items() if k == own or not k.startswith("early_")}

    def expert_view(self, xs):
        # the late expert sees the middle early-season indicator
        return self.early_view(xs, EARLY_THRESHOLDS[1])

    def rest_days(self, team, date):
        if team not in self.last_game:
            return MAX_REST_DAYS
        return min((date - self.last_game[team]).days, MAX_REST_DAYS)

    @staticmethod
    def sides(pre):
        # (favourite, underdog)
        return (pre["home"], pre["away"]) if pre["whos_favored"] == "home" else (pre["away"], pre["home"])

    def short_rest(self, pre):
        # the favourite's and the underdog's rest days, capped at 4
        return tuple(min(self.rest_days(team, pre["date"]), 4) for team in self.sides(pre))

    @staticmethod
    def holiday(date):
        # New Year's Day or Martin Luther King Day (third Monday of January)
        if (date.month, date.day) == (1, 1):
            return True
        return date.month == 1 and date.weekday() == 0 and 15 <= date.day <= 21

    def series_game(self, pre):
        # playoff series game number (capped), a fixed code for the other non-regular games (play-in, Cup final), 0 in the regular season
        if pre["playoffs"]:
            return float(min(self.series[frozenset((pre["home"], pre["away"]))] + 1, SERIES_CAP))
        return 0.0 if pre["regular"] else PLAY_IN_SERIES

    def late_rank(self, team):
        # late season (60+ team games): conference rank by wins minus losses, else None
        wins, losses = self.record[team]
        if wins + losses < STAKES_FROM:
            return None
        mates = [(w, l) for t, (w, l) in self.record.items() if t != team and CONFERENCE.get(t) == CONFERENCE.get(team)]
        return 1 + sum(w - l > wins - losses for w, l in mates)

    def bubble(self, team):
        # a playoff-bubble team (seeds 7-10)
        rank = self.late_rank(team)
        return rank is not None and 7 <= rank <= 10

    def lottery(self, team):
        # out of the playoff picture (below seed 8, seed 10 since the play-in; before it, seeds 9-10 are also bubble teams)
        rank = self.late_rank(team)
        return rank is not None and rank > (10 if self.season >= 2021 else 8)

    def final_game(self, team):
        # the team's last regular-season game
        return sum(self.record[team]) == SEASON_LENGTH.get(self.season, 82) - 1

    def contender(self, team):
        # a .600+ team with 41+ games played (stars' minutes are managed for the playoffs)
        wins, losses = self.record[team]
        return wins + losses >= CONTENDER_GAMES and wins / (wins + losses) >= CONTENDER_PCT

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

    def phase(self, pre, target, late_left=LATE_PREDICT_LEFT):
        if pre["playoffs"]:
            return "playoff"
        if pre["regular"] and target == "id_total" and self.season_games > 15 * SEASON_LENGTH.get(pre["season"], 82) - late_left:  # 30 teams: 15 league games per team game
            return "late"
        return "early" if self.season_games < EARLY_THRESHOLDS[1] else "main"

    def predict(self, pre):
        if self.ridge_season != pre["season"]:
            self.fit_ridges(pre["season"])
        self.pending[self.game_key(pre)] = {"logit": {}, "routed": {}, "playoff": {}, "segment": {}}
        x = self.features(pre)
        out = {}
        for target, models in self.models.items():
            xs = self.scalers[target].transform_one(self.view(x, target))
            ps = [model.predict_proba_one(self.early_view(xs, g)).get(1, 0.5) for g, model in models]
            out[target] = sum(ps) / len(ps)
            if self.phase(pre, target) == "late":  # late regular-season total games: 50/50 blend with the late expert
                out[target] = 0.5 * out[target] + 0.5 * self.late_model.predict_proba_one(self.expert_view(xs)).get(1, 0.5)
        spread_logit = logit(out["id_spread"])  # the model's own spread lean (regular season: before segments and routers)
        if pre["playoffs"] and pre["spread"] is not None and pre["total"] is not None:
            self.playoff_lines(pre, x, out)
        if pre["regular"] and pre["spread"] is not None and pre["total"] is not None:
            self.segment_lines(pre, x, out)
        self.route(pre, out, spread_logit)
        return self.calibrate(pre, out)

    def playoff_context(self, pre, x):
        # playoff context per target: line, series game, round, rest and the series' previous game -
        # spread: venue, rest edge, the favourite's line error and the total residual there;
        # total: its regulation total residual, overtime points and the total line move since
        fav, dog = self.sides(pre)
        played = self.playoff_opponents[pre["home"]]
        rnd = min(len(played) + (pre["away"] not in played), 4)
        rest_fav, rest_dog = self.short_rest(pre)
        last = self.series_last.get(frozenset((fav, dog)))
        # the series' previous game: the favourite's line error, the total residual, overtime points and the total line move since (0 in game 1)
        fav_error, total_error, overtime, line_move = (last[0][fav], last[1], last[2], pre["total"] - last[3]) if last else (0.0, 0.0, 0.0, 0.0)
        rounds = [float(rnd == 2), float(rnd == 3), float(rnd == 4)]
        return {"id_spread": [x["fav_home"], x["spread"], x["total_centered"], x["series_game"], *rounds, min(rest_fav, rest_dog), rest_fav - rest_dog,
                              fav_error, total_error],
                "id_total": [x["total_centered"], x["spread"], x["series_game"], *rounds, min(rest_fav, rest_dog), total_error - overtime, overtime, line_move]}

    def fit_ridges(self, season):
        # once per season: the line-error ridges (playoffs per target, regular-season segments) on all previous seasons' games
        self.ridge_season = season
        for t in TARGETS:
            self.playoff_reg[t] = fit_line_ridge([(f, y) for s, f, y in self.playoff_rows[t] if s < season])
        for seg in SEGMENTS:
            self.segment_reg[seg] = fit_line_ridge([(f, y) for s, f, y in self.segment_rows[seg] if s < season])

    def playoff_lines(self, pre, x, out):
        # playoff games: P(1) = Phi(predicted line error / residual sd) of the playoff regressions (spread: replaces the model; total: 50/50)
        for t, f in self.playoff_context(pre, x).items():
            self.pending[self.game_key(pre)]["playoff"][t] = f
            if self.playoff_reg[t] is not None:
                p = line_ridge_prob(self.playoff_reg[t], f)
                out[t] = p if t == "id_spread" else 0.5 * out[t] + 0.5 * p

    def segments(self, pre, x):
        # the game's regular-season segments, in SEGMENTS order: the visitor on a back-to-back at altitude; a rest mismatch (one team
        # on a back-to-back, the other with 2+ more days of rest); a team on a big ATS streak; a team whose previous game went to
        # overtime; a contending favourite on the second night of a back-to-back (load management); both teams met the same
        # opponent within the last week (the market may overrate the gap between their results there).
        # Each with its ridge context: line, venue, total deviation from the league level and rest (spread: per side, plus both ATS
        # streaks in ats_streak, the gap vs the common opponent in co_week; total: the favourite's rest edge - the shorter rest is
        # always 1 in rest_mismatch)
        fav, dog = self.sides(pre)
        rest_fav, rest_dog = self.short_rest(pre)
        streaks = [self.ats_streak(pre, fav), self.ats_streak(pre, dog)]
        fired = {"altitude_b2b": pre["home"] in ALTITUDE and self.rest_days(pre["away"], pre["date"]) <= 1,
                 "rest_mismatch": abs(rest_fav - rest_dog) >= 2 and min(rest_fav, rest_dog) <= 1,
                 "ats_streak": any(streaks),
                 "prev_ot": self.last_ot.get((pre["season"], fav)) or self.last_ot.get((pre["season"], dog)),
                 "contender_b2b": self.contender(fav) and rest_fav <= 1,
                 "co_week": self.common_gap(pre, fav, dog) is not None}
        line = [x["fav_home"], x["spread"], x["total_centered"]]
        context = {"rest_mismatch": line + [rest_fav - rest_dog], "ats_streak": line + [rest_fav, rest_dog, *streaks],
                   "co_week": line + [rest_fav, rest_dog, self.common_gap(pre, fav, dog) or 0.0]}
        return {seg: context.get(seg, line + [rest_fav, rest_dog]) for seg in SEGMENTS if fired[seg]}

    def common_gap(self, pre, fav, dog):
        # the latest opponent both teams met within COMMON_DAYS: the favourite's line error there minus the underdog's, else None
        seen = {opp: e for d, opp, e in self.recent[dog] if (pre["date"] - d).days <= COMMON_DAYS}
        for d, opp, e in reversed(self.recent[fav]):
            if (pre["date"] - d).days > COMMON_DAYS:
                return None
            if opp in seen and opp != dog:
                return e - seen[opp]
        return None

    def ats_streak(self, pre, team):
        # the sum of the team's last line errors this season when they share a sign and are big (an ATS run the market may overprice), else 0
        e = self.line_errors[(pre["season"], team)][-ATS_STREAK_GAMES:]
        if len(e) == ATS_STREAK_GAMES and (all(v > 0 for v in e) or all(v < 0 for v in e)) and sum(abs(v) for v in e) >= ATS_STREAK_GAMES * ATS_STREAK_MEAN:
            return sum(e)
        return 0.0

    def segment_lines(self, pre, x, out):
        # segment games: P(1) = Phi(predicted line error / residual sd) of the segment's ridge, fit once per season on previous
        # seasons' segment games, replacing or blended with the model for that target only; the context ends with the model's
        # own logit, so each ridge also learns how far to trust the model in its segment
        model_logit = {t: logit(p) for t, p in out.items()}
        for seg, context in self.segments(pre, x).items():
            t, weight = SEGMENTS[seg]
            f = context + [model_logit[t]]
            self.pending[self.game_key(pre)]["segment"][seg] = f
            if self.segment_reg[seg] is not None:
                out[t] = (1 - weight) * out[t] + weight * line_ridge_prob(self.segment_reg[seg], f)

    def ats_form(self, pre, team):
        # the team's mean line error over its last 5 regular-season games (0 before its first)
        errors = self.line_errors[(pre["season"], team)][-5:]
        return sum(errors) / len(errors) if errors else 0.0

    def route(self, pre, out, spread_logit):
        # override rules (target -> (rule, side)): the output is the rule's running hit rate for its side, so a rule bets on whichever
        # side its record favours; spread - the underdog when the model leans to a favourite whose ATS form is not far below the
        # underdog's, else the lone playoff-bubble team; total - a team's last regular-season game (over), two lottery teams (the
        # running over rate has stayed below .5 since 2010: in practice an under rule)
        rules = {}
        if pre["regular"] and pre["season"] == self.season:  # this season's state (standings, ATS form) starts with its first result
            fav, dog = self.sides(pre)
            if spread_logit > FAV_LEAN and self.ats_form(pre, fav) - self.ats_form(pre, dog) > FAV_LEAN_ATS_GAP:
                rules["id_spread"] = ("fav_lean", -1)  # the model's favourite leans over-trust the favourite
            elif self.bubble(fav) != self.bubble(dog):
                rules["id_spread"] = ("bubble", 1 if self.bubble(fav) else -1)
            if self.final_game(fav) or self.final_game(dog):
                rules["id_total"] = ("final", 1)
            elif self.lottery(fav) and self.lottery(dog):
                rules["id_total"] = ("lottery", 1)
        for t, (rule, side) in rules.items():
            self.pending[self.game_key(pre)]["routed"][t] = (rule, side)
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
        for t in TARGETS:
            z = logit(out[t])
            tk = (t, self.phase(pre, t))
            self.pending[self.game_key(pre)]["logit"][t] = (z, tk)
            out[t] = self.calibrated(z, tk)
        return out

    def learn_temperature(self, logits, result):
        for t, (z, tk) in logits.items():
            if result[t] in (0, 1):
                p = self.calibrated(z, tk)
                self.temperature[tk] = max(TEMPERATURE_MIN, self.temperature[tk] + TEMPERATURE_LR * (result[t] - p) * z)

    def learn(self, pre, result):
        pending = self.pending.pop(self.game_key(pre))
        self.learn_temperature(pending["logit"], result)
        for t, (rule, side) in pending["routed"].items():
            if result[t] in (0, 1):
                self.route_rate[rule][0] += float((result[t] == 1) == (side > 0))
                self.route_rate[rule][1] += 1
        self.learn_models(pre, result)
        home_error = self.record_line_errors(pre, result, pending)
        self.update_state(pre, result, home_error)

    def learn_models(self, pre, result):
        # the ensembles and the late total expert, on features scaled before this row updates the scaler
        x = self.features(pre)
        for target, models in self.models.items():
            y = result[target]
            if y in (0, 1):
                xt = self.view(x, target)
                xs = self.scalers[target].transform_one(xt)
                if target == "id_spread" and self.season_games <= ELO_WARMUP:
                    xt = {k: v for k, v in xt.items() if k not in ELO_FEATURES}  # zeroed Elo features do not shape the scaler
                self.scalers[target].learn_one(xt)
                for g, model in models:
                    learn_smoothed(model, self.early_view(xs, g), y)
                if self.phase(pre, target, LATE_LEARN_LEFT) == "late":  # the late total expert learns in a longer window
                    learn_smoothed(self.late_model, self.expert_view(xs), y)

    def record_line_errors(self, pre, result, pending):
        # line errors (favourite's margin minus the line, total residual) as ridge rows and per-team history; returns the home team's error
        margin = result["score_home"] - result["score_away"]
        errors = {"id_spread": (margin if pre["whos_favored"] == "home" else -margin) - (pre["spread"] or 0.0),
                  "id_total": result["score_home"] + result["score_away"] - (pre["total"] or 0.0)}
        for t, f in pending["playoff"].items():
            self.playoff_rows[t].append((pre["season"], f, errors[t]))
        for seg, f in pending["segment"].items():
            self.segment_rows[seg].append((pre["season"], f, errors[SEGMENTS[seg][0]]))
        home_error = errors["id_spread"] if pre["whos_favored"] == "home" else -errors["id_spread"]  # the home team's margin minus its line-implied margin
        if pre["regular"] and pre["spread"] is not None and pre["total"] is not None:
            self.line_errors[(pre["season"], pre["home"])].append(home_error)
            self.line_errors[(pre["season"], pre["away"])].append(-home_error)
            for team in (pre["home"], pre["away"]):
                self.last_ot[(pre["season"], team)] = result["ot_home"] + result["ot_away"] > 0
            self.recent[pre["home"]].append((pre["date"], pre["away"], home_error))
            self.recent[pre["away"]].append((pre["date"], pre["home"], -home_error))
        return home_error

    def update_state(self, pre, result, home_error):
        if pre["season"] != self.season:
            self.season = pre["season"]
            self.season_games = 0
            self.series = defaultdict(int)
            self.series_last = {}
            self.playoff_opponents = defaultdict(set)
            self.record = defaultdict(lambda: [0, 0])
            for t in list(self.elo):
                self.elo[t] = (1 - ELO_REGRESS) * self.elo[t] + ELO_REGRESS * ELO_START
        if pre["playoffs"]:
            self.series[frozenset((pre["home"], pre["away"]))] += 1
            if pre["spread"] is not None and pre["total"] is not None:
                home, away = pre["home"], pre["away"]
                overtime = result["ot_home"] + result["ot_away"]
                self.series_last[frozenset((home, away))] = ({home: home_error, away: -home_error}, result["score_home"] + result["score_away"] - pre["total"],
                                                             overtime, pre["total"])
                self.playoff_opponents[home].add(away)
                self.playoff_opponents[away].add(home)
        elif pre["regular"]:
            self.season_games += 1  # the league's regular-season clock (the Cup final, play-in and playoffs do not advance it)
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
        hca = ELO_HOME_ADV + (ELO_EARLY_HCA if early else 0)
        expected_home = 1 / (1 + 10 ** ((self.elo[away] - self.elo[home] - hca) / 400))
        k = ELO_CATCHUP_K if self.season_games <= ELO_WARMUP + ELO_CATCHUP_GAMES else ELO_K
        delta = k * (int(home_won) - expected_home)
        if early and abs(self.elo[home] - self.elo[away]) < ELO_TOSSUP_GAP:
            delta *= ELO_TOSSUP_BOOST
        self.elo[home] += delta
        self.elo[away] -= delta


if __name__ == "__main__":
    print_summary(evaluate(Model()))
