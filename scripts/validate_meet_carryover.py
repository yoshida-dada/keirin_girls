"""日跨ぎ特徴の純増検証: 同一開催・直前日の着順を特徴に足すと着順予測が上がるか。

初日→二日目の巻き返し(初日4-6着=二日目+6.2pt/3着内=-6.8pt, analyze_meet_day)は実在。だが残差の
一部は勝ち上がり(楽/強い組)でモデルが対戦相手経由で既に捕捉している可能性。ここでは
「その選手の同一開催・直前日の着順」を as-of 特徴として着順モデルに追加し、baselineを
walk-forward(out-of-sample)で純増するかを見る。純増すれば"モデルが取りこぼす日跨ぎ調子"が実在。

特徴(3列, レース内非相対のリーク無し): p_top3 / p_mid46 / p_bad7（直前日 3着内 / 4-6着 / 7着以下。
全0=初日 or 同一開催に前日出走なし）。GBDTが巻き返しの非単調性を学べる形。

  PYTHONIOENCODING=utf-8 python scripts/validate_meet_carryover.py           # 男子
"""
from __future__ import annotations

import argparse
import copy
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.rider_narabi import compute_narabi_features
from src.model.feature_sets import men_features, girls_features
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS, DEFAULT_PARAMS
from src.backtest.walkforward import fold_boundaries


def _names_and_finishes(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    names = {}
    for rid, car, nm in c.execute("SELECT race_id,car_number,rider_name FROM entries"):
        names[(rid, car)] = nm
    # (meet,name) -> {day: pos}
    mf = defaultdict(dict)
    for rid, nm, pos in c.execute("SELECT race_id,rider_name,position FROM results WHERE position IS NOT NULL"):
        mf[(rid[:10], nm)].setdefault(int(rid[10:12]), pos)
    c.close()
    return names, mf


def _prior_pos(meet, name, day, mf):
    """同一開催で day より前の最新日の着順。無ければ None。"""
    dd = mf.get((meet, name))
    if not dd:
        return None
    priors = [(d, p) for d, p in dd.items() if d < day]
    return max(priors, key=lambda x: x[0])[1] if priors else None


def _carry_cols(samples, names, mf):
    out = []
    for s in samples:
        meet, day = s.race_id[:10], int(s.race_id[10:12])
        rows = []
        for car in s.car_numbers:
            nm = names.get((s.race_id, car))
            pp = _prior_pos(meet, nm, day, mf) if nm else None
            rows.append([1.0 if pp is not None and pp <= 3 else 0.0,
                         1.0 if pp is not None and 4 <= pp <= 6 else 0.0,
                         1.0 if pp is not None and pp >= 7 else 0.0])
        s2 = copy.copy(s)
        s2.X = np.hstack([s.X, np.array(rows, dtype=float)])
        s2.feature_names = list(s.feature_names) + ["p_top3", "p_mid46", "p_bad7"]
        out.append(s2)
    return out


def _tri10(dist, o3):
    return int(tuple(o3) in [k for k, _ in sorted(dist.items(), key=lambda kv: -kv[1])[:10]])


def _s2t3(dist, a2):
    marg = defaultdict(float)
    for (a, b, c), p in dist.items():
        marg[b] += p
    return int(a2 in [x for x, _ in sorted(marg.items(), key=lambda kv: -kv[1])[:3]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--men", action="store_true", default=True)
    ap.add_argument("--girls", dest="men", action="store_false")
    ap.add_argument("--db")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    db = args.db or str(DATA_DIR / ("keirin_men.sqlite" if args.men else "keirin.sqlite"))
    feats = men_features() if args.men else girls_features()
    params = MEN_PARAMS if args.men else DEFAULT_PARAMS

    base = load_samples(db, field_size=([7, 9] if args.men else 7), features=PL_FEATURES_FULL)
    samples = augment_samples(base, db, feats)
    narabi = compute_narabi_features(db)
    names, mf = _names_and_finishes(db)
    # 前日出走ありのレース割合（参考）
    withprior = sum(1 for s in samples if int(s.race_id[10:12]) >= 2)
    print(f"{'男子' if args.men else 'ガールズ'} {len(samples)}レース / 二日目以降 {withprior}\n")
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    agg = {k: 0 for k in ("n", "b_t1", "a_t1", "b_s2", "a_s2", "b_t10", "a_t10", "b_ll", "a_ll")}
    perfold = []
    for a, b, c in bounds:
        tr, te = samples[a:b], samples[b:c]
        bm = train_gbdt(tr)
        am = train_gbdt(_carry_cols(tr, names, mf))
        te_a = _carry_cols(te, names, mf)
        f = {k: 0 for k in agg}
        for s, sa in zip(te, te_a):
            if len(s.order) < 3:
                continue
            o3 = tuple(s.order[:3])
            npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
            stb = bm.strengths(s.X, s.car_numbers)
            sta = am.strengths(sa.X, sa.car_numbers)
            db_ = corrected_trifecta_probs(stb, npos, params)
            da_ = corrected_trifecta_probs(sta, npos, params)
            f["b_t1"] += int(max(stb, key=stb.get) == o3[0]); f["a_t1"] += int(max(sta, key=sta.get) == o3[0])
            f["b_s2"] += _s2t3(db_, o3[1]); f["a_s2"] += _s2t3(da_, o3[1])
            f["b_t10"] += _tri10(db_, o3); f["a_t10"] += _tri10(da_, o3)
            pb = db_.get(o3, 0.0); f["b_ll"] += -math.log(pb) if pb > 0 else 50.0
            pa = da_.get(o3, 0.0); f["a_ll"] += -math.log(pa) if pa > 0 else 50.0
            f["n"] += 1
        for k in agg:
            agg[k] += f[k]
        perfold.append(f)

    n = agg["n"]
    print(f"【日跨ぎ特徴(直前日着順)の純増】 out-of-sample {n}レース  baseline → +carryover")
    def line(lbl, bk, ak, pct=True):
        bv, av = agg[bk] / n, agg[ak] / n
        if pct:
            print(f"  {lbl:<14} {bv*100:6.2f}% → {av*100:6.2f}%  (Δ{(av-bv)*100:+.2f})")
        else:
            print(f"  {lbl:<14} {bv:7.4f} → {av:7.4f}  (Δ{av-bv:+.4f}{' 改善' if av<bv else ''})")
    line("1着 top1", "b_t1", "a_t1")
    line("2着 top3", "b_s2", "a_s2")
    line("三連単 top10", "b_t10", "a_t10")
    line("三連単 log-loss", "b_ll", "a_ll", pct=False)
    print("\n  per-fold Δ(+carryover−baseline):")
    for i, f in enumerate(perfold):
        m = f["n"] or 1
        print(f"    fold{i} n={f['n']}: top1 {(f['a_t1']-f['b_t1'])/m*100:+.2f} / "
              f"2着top3 {(f['a_s2']-f['b_s2'])/m*100:+.2f} / tri10 {(f['a_t10']-f['b_t10'])/m*100:+.2f} / "
              f"ll {(f['a_ll']-f['b_ll'])/m:+.4f}")


if __name__ == "__main__":
    main()
