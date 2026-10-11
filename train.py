# -*- coding: utf-8 -*-
"""
NBA spread ve total: hedef başına bir River pipeline.

GameFeatures maç öncesi kolonlardan özellik üretir; durumu (son maç tarihi,
lig toplam çizgisi) sonuç açılan maçlarla günceller.

River Pipeline.learn_one, transformer'ın learn_one'ını transform_one'dan ÖNCE
çağırır. Durum orada hemen güncellenirse öğrenilen özellikler maçın kendisini
görür (ör. rest_edge hep 0 olur) ve tahmindekinden farklılaşır. Bu yüzden
learn_one maçı yalnızca bekletir; bekleyen maçlar, tarihi daha ileri bir maç
dönüştürülürken duruma işlenir. Böylece bir maç tahminde de öğrenmede de aynı
durumla, yani yalnızca önceki günlerin maçlarıyla dönüştürülür.
"""

from river import base, compose, linear_model, optim, preprocessing

from prepare import print_summary, progressive_score

MAX_REST_DAYS = 20
LR = 0.005
LINE_ALPHA = 0.01  # lig toplam çizgisinin EWMA hızı

# spread: çizgi ve dinlenme farkı. total: çizgi seviyesi ve bağlam.
FEATURES = {
    "id_spread": ["fav_home", "spread", "rest_edge"],
    "id_total": ["fav_home", "spread", "total", "total_centered", "playoffs"],
}


class GameFeatures(base.Transformer):
    def __init__(self):
        self.last_game = {}
        self.mean_total = None
        self.pending = []  # sonucu açılmış ama duruma henüz işlenmemiş maçlar

    def rest_days(self, team, date):
        previous = self.last_game.get(team)
        if previous is None:
            return MAX_REST_DAYS
        return min((date - previous).days, MAX_REST_DAYS)

    def transform_one(self, x):
        home, away, date = x["home"], x["away"], x["date"]
        self.apply_pending(date)
        fav_home = x["whos_favored"] == "home"
        sign = 1.0 if fav_home else -1.0
        total = float(x["total"] or 0.0)
        level = total if self.mean_total is None else self.mean_total
        return {
            "fav_home": float(fav_home),
            "spread": float(x["spread"] or 0.0),
            "total": total,
            "total_centered": total - level,
            "rest_edge": sign * (self.rest_days(home, date) - self.rest_days(away, date)),
            "playoffs": float(bool(x["playoffs"])),
        }

    def learn_one(self, x):
        self.pending.append(x)

    def apply_pending(self, date):
        """date'ten önceki günlerin bekleyen maçlarını sırayla duruma işle."""
        ready = [x for x in self.pending if x["date"] < date]
        self.pending = [x for x in self.pending if x["date"] >= date]
        for x in ready:
            self.update_state(x)

    def update_state(self, x):
        self.last_game[x["home"]] = x["date"]
        self.last_game[x["away"]] = x["date"]
        if x["total"]:
            total = float(x["total"])
            if self.mean_total is None:
                self.mean_total = total
            else:
                self.mean_total += LINE_ALPHA * (total - self.mean_total)


def make_model(target):
    return (
        GameFeatures()
        | compose.Select(*FEATURES[target])
        | preprocessing.StandardScaler()
        | linear_model.LogisticRegression(optim.SGD(LR))
    )


if __name__ == "__main__":
    print_summary(progressive_score(make_model))
