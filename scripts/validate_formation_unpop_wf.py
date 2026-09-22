"""「▲以降がモデル近差なら人気薄」の軸堅+7〜8ptを as-of walk-forward で確定(男子7車)。

in-sampleリーク排除: 各foldでtestより過去だけ lambdarank 再学習し、そのas-ofモデルで
印(◎○▲△)・レースタイプ・near-tie を判定。三連単12点/二車単6点で 素(モデル順) vs 人気薄10%
の回収率を、軸堅/全体でfold別+統合、差分(人気薄−素)のブートストラップ95%区間まで出す。
市場人気=三連単確定オッズ逆算。

  PYTHONIOENCODING=utf-8 python scripts/validate_formation_unpop_wf.py --folds 6
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.model.train_gbdt import train_gbdt
from src.model.race_type import classify_race
from src.ev.market import implied_trifecta_probs
from src.backtest.walkforward import fold_boundaries

TIE = 0.10


def _load(db, rids):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    tri = {rid: (tuple(int(x) for x in cb.split("-")), p)
           for rid, cb, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    exa = {rid: (tuple(int(x) for x in cb.split("-")), p)
           for rid, cb, p in c.execute("SELECT race_id,combo,payout FROM payouts_exacta")}
    odds = defaultdict(dict)
    for rid, cb, o in c.execute("SELECT race_id,combo,odds FROM odds_final_trifecta"):
        if rid in rids:
            odds[rid][tuple(int(x) for x in cb.split("-"))] = o
    c.close()
    return tri, exa, odds


def _pop(orace):
    q = implied_trifecta_probs(orace)
    pop = defaultdict(float)
    for (a, b, c), v in q.items():
        pop[a] += v; pop[b] += v; pop[c] += v
    tot = sum(pop.values())
    return {k: v / tot for k, v in pop.items()} if tot else {}


def _marks(st, pop, tie):
    ranked = sorted(st, key=st.get, reverse=True)
    pool = ranked[2:6]
    if tie > 0 and pop:
        pool = sorted(pool, key=lambda car: (
            sum(1 for o in pool if st[o] > st[car] * (1 + tie)), pop.get(car, 1.0)))
    return ranked[:2] + pool[:2]


def _sets(marks):
    a12, rest = marks[:2], marks[:4]
    ts = {p for p in permutations(rest, 3) if p[0] in a12}
    es = {p for p in permutations(rest, 2) if p[0] in a12}
    return ts, es


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()
    model, _, _ = load_for(False)
    base = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    rids = {s.race_id for s in samples}
    tri, exa, odds = _load(args.db, rids)
    samples = [s for s in samples if s.race_id in tri and s.race_id in exa and s.race_id in odds]
    samples.sort(key=lambda s: (s.date, s.race_id))
    n = len(samples)
    bounds = fold_boundaries(n, n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"男子7車 {samples[0].date}〜{samples[-1].date}  {n}レース  as-of walk-forward "
          f"{len(bounds)}fold  素 vs 人気薄{int(TIE*100)}%\n")

    # rec[(bucket,kind)] = list per race of (base_pts,base_ret,base_hit, up_pts,up_ret,up_hit, fold)
    rec = defaultdict(list)
    for fi, (a, b, c) in enumerate(bounds):
        m = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            st = m.strengths(s.X, s.car_numbers)
            if len(st) < 4:
                continue
            pop = _pop(odds[s.race_id])
            bk = classify_race(st).label
            mb = _marks(st, pop, 0.0)
            mu = _marks(st, pop, TIE)
            tb, eb = _sets(mb)
            tu, eu = _sets(mu)
            for kind, (sb, su, pay) in (("三連単", (tb, tu, tri[s.race_id])),
                                        ("二車単", (eb, eu, exa[s.race_id]))):
                combo, py = pay
                combo = tuple(combo)
                row = (len(sb), py if combo in sb else 0, int(combo in sb),
                       len(su), py if combo in su else 0, int(combo in su), fi)
                rec[(bk, kind)].append(row)
                rec[("全体", kind)].append(row)

    def roi(rows, base):
        i0, i1 = (0, 1) if base else (3, 4)
        pts = sum(r[i0] for r in rows); ret = sum(r[i1] for r in rows)
        return ret / (pts * 100) * 100 if pts else 0

    def hitr(rows, base):
        i = 2 if base else 5
        return sum(r[i] for r in rows) / len(rows) * 100 if rows else 0

    def boot_delta(rows, B=3000, seed=1):
        if len(rows) < 30:
            return 0, 0
        rng = np.random.default_rng(seed)
        bp = np.array([r[0] for r in rows]); br = np.array([r[1] for r in rows])
        up = np.array([r[3] for r in rows]); ur = np.array([r[4] for r in rows])
        m = len(rows); out = []
        for _ in range(B):
            idx = rng.integers(0, m, m)
            rb = br[idx].sum() / (bp[idx].sum() * 100) * 100
            ru = ur[idx].sum() / (up[idx].sum() * 100) * 100
            out.append(ru - rb)
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    for kind in ("三連単", "二車単"):
        for bk in ("全体", "軸堅"):
            rows = rec.get((bk, kind))
            if not rows:
                continue
            print(f"━━ {kind} / {bk}  {len(rows)}レース ━━")
            print(f"   {'fold':>4}{'R数':>6}{'素ROI':>9}{'人気薄ROI':>11}{'差':>8}")
            wins = 0
            for fi in range(len(bounds)):
                fr = [r for r in rows if r[6] == fi]
                if not fr:
                    continue
                rb, ru = roi(fr, True), roi(fr, False)
                wins += ru > rb
                print(f"   {fi:>4}{len(fr):>6}{rb:>8.1f}%{ru:>10.1f}%{ru-rb:>+7.1f}")
            rb, ru = roi(rows, True), roi(rows, False)
            lo, hi = boot_delta(rows)
            print(f"   {'統合':>4}{len(rows):>6}{rb:>8.1f}%{ru:>10.1f}%{ru-rb:>+7.1f}")
            print(f"        的中 素{hitr(rows,True):.1f}% / 人気薄{hitr(rows,False):.1f}%  "
                  f"人気薄が勝ったfold {wins}/{len(bounds)}")
            print(f"        差分ブート95%区間 [{lo:+.1f} 〜 {hi:+.1f}]pt  "
                  f"差>0有意={'YES' if lo > 0 else 'NO'}\n")


if __name__ == "__main__":
    main()
