"""◎○→◎○▲△→◎○▲△(三連単12点) と ◎○→◎○▲△(二車単6点) の回収率、
   および「▲以降がモデルで近差なら人気薄を選ぶ」変種のシミュレーション(男子7車)。

印=本番モデルの勝率順位(◎1/○2/▲3/△4)。◎○は固定。▲△の枠を、
モデル確率が相対TIE%以内で近差の候補群の中では市場人気の薄い車から埋める(TIE=0で素のモデル順)。
市場人気=三連単確定オッズ逆算(implied_trifecta_probs)の全着順マージナル share。
実払戻(payouts_trifecta/exacta)で決済。TIEを 0/0.10/0.20/0.35 でスイープし全体/レースタイプ別に集計。

  PYTHONIOENCODING=utf-8 python scripts/validate_formation_unpop.py --days 365
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.model.race_type import classify_race
from src.ev.market import implied_trifecta_probs


def _adate(rid):
    return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=int(rid[10:12]) - 1)


def _load(db, rids):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    tri = {rid: (tuple(int(x) for x in combo.split("-")), p)
           for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    exa = {rid: (tuple(int(x) for x in combo.split("-")), p)
           for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_exacta")}
    odds = defaultdict(dict)
    for rid, combo, o in c.execute("SELECT race_id,combo,odds FROM odds_final_trifecta"):
        if rid in rids:
            odds[rid][tuple(int(x) for x in combo.split("-"))] = o
    c.close()
    return tri, exa, odds


def _market_pop(odds_race):
    """三連単オッズ→全着順マージナルの市場人気 share {car: pop}。"""
    q = implied_trifecta_probs(odds_race)
    pop = defaultdict(float)
    for (a, b, c), v in q.items():
        pop[a] += v; pop[b] += v; pop[c] += v
    tot = sum(pop.values())
    return {k: v / tot for k, v in pop.items()} if tot else {}


def _marks(st, pop, tie):
    """勝率順に◎○▲△。▲以降はモデル近差(相対tie%以内)クラスタ内で人気薄(pop昇順)を優先。"""
    ranked = sorted(st, key=st.get, reverse=True)
    top2 = ranked[:2]                       # ◎○ 固定
    pool = ranked[2:6]                       # ▲△候補(rank3..6)
    if tie > 0 and pop:
        def tie_rank(car):                  # 自分より確率が tie% 超で上回る候補数=近差クラスタID
            return sum(1 for o in pool if st[o] > st[car] * (1 + tie))
        pool = sorted(pool, key=lambda car: (tie_rank(car), pop.get(car, 1.0)))
    return top2 + pool[:2]                   # ◎○▲△


def _tri_set(marks):
    a12, rest = marks[:2], marks[:4]
    return {p for p in permutations(rest, 3) if p[0] in a12}


def _exa_set(marks):
    a12, rest = marks[:2], marks[:4]
    return {p for p in permutations(rest, 2) if p[0] in a12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()
    model, _, _ = load_for(False)
    base = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    rids = {s.race_id for s in samples}
    tri, exa, odds = _load(args.db, rids)
    dated = [(s, _adate(s.race_id)) for s in samples
             if s.race_id in tri and s.race_id in exa and s.race_id in odds]
    days = sorted({d for _, d in dated})[-args.days:]
    dayset = set(days)
    target = [s for s, d in dated if d in dayset]
    print(f"男子7車 直近{args.days}日 {days[0]}〜{days[-1]}  対象{len(target)}レース"
          f"(三連単オッズ有)\n")

    ties = [0.0, 0.10, 0.20, 0.35]
    # agg[(tie, bucket, kind)] = {pts, ret, hit, n}
    agg = defaultdict(lambda: {"pts": 0, "ret": 0, "hit": 0, "n": 0})
    for s in target:
        rid = s.race_id
        st = model.strengths(s.X, s.car_numbers)
        if len(st) < 4:
            continue
        pop = _market_pop(odds[rid])
        bucket = classify_race(st).label
        tcombo, tpay = tri[rid]; ecombo, epay = exa[rid]
        for tie in ties:
            marks = _marks(st, pop, tie)
            ts, es = _tri_set(marks), _exa_set(marks)
            for bk in (bucket, "全体"):
                a = agg[(tie, bk, "三連単")]
                a["n"] += 1; a["pts"] += len(ts)
                if tuple(tcombo) in ts:
                    a["hit"] += 1; a["ret"] += tpay
                b = agg[(tie, bk, "二車単")]
                b["n"] += 1; b["pts"] += len(es)
                if tuple(ecombo) in es:
                    b["hit"] += 1; b["ret"] += epay

    def roi(d): return d["ret"] / (d["pts"] * 100) * 100 if d["pts"] else 0
    for kind in ("三連単", "二車単"):
        print(f"━━━ {kind}  (◎○→◎○▲△{'→◎○▲△' if kind=='三連単' else ''}) ━━━")
        for bk in ("全体", "軸堅", "標準", "混戦"):
            base_d = agg.get((0.0, bk, kind))
            if not base_d or base_d["n"] == 0:
                continue
            n = base_d["n"]
            print(f"■ {bk}  {n}レース")
            print(f"   {'TIE幅':<8}{'点/R':>7}{'的中率':>9}{'回収率':>9}{'素比':>8}")
            base_roi = roi(base_d)
            for tie in ties:
                d = agg[(tie, bk, kind)]
                r = roi(d)
                tag = "素(モデル順)" if tie == 0 else f"人気薄{int(tie*100)}%"
                print(f"   {tag:<10}{d['pts']/n:>6.1f}{d['hit']/n*100:>8.1f}%{r:>8.1f}%"
                      f"{r-base_roi:>+7.1f}")
            print()


if __name__ == "__main__":
    main()
