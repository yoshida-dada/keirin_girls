"""二車単2〜3点の選定ポリシーを as-of walk-forward で比較し、回収率が安定する型を選ぶ(男子7車)。

過去の二車単オッズは未収集のため、三連単確定オッズ(odds_final_trifecta)のimplied確率を
1-2着へ周辺化して「合成二車単オッズ」を作り(控除は勝ち払戻平均で1回校正)、選定にのみ使う。
決済は実際の二車単払戻(payouts_exacta)。モデルは各foldで過去だけ再学習(リーク排除)。
モデル二車単確率=himo補正三連単の周辺化。ユーザー要望「三連単で的中を担保、二車単は穴に振ってよい。
数パターンで回収率が安定するものを選定」に対応し、本命寄り〜穴寄りを横断して比較する。

  PYTHONIOENCODING=utf-8 python scripts/validate_exacta_selection.py --folds 6
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.model.train_gbdt import train_gbdt
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.ev.market import implied_trifecta_probs
from src.backtest.walkforward import fold_boundaries


def _load(db, rids):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    exa = {rid: (tuple(int(x) for x in cb.split("-")), p)
           for rid, cb, p in c.execute("SELECT race_id,combo,payout FROM payouts_exacta")}
    odds = defaultdict(dict)
    for rid, cb, o in c.execute("SELECT race_id,combo,odds FROM odds_final_trifecta"):
        if rid in rids:
            odds[rid][tuple(int(x) for x in cb.split("-"))] = o
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    c.close()
    lines = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        lines[rid] = [x for x in ls if x]
    return exa, odds, lines


def _model_exacta(st, ln):
    tri = corrected_trifecta_probs(st, {}, MEN_PARAMS, lines=ln)
    ex = defaultdict(float)
    for (a, b, c), p in tri.items():
        ex[(a, b)] += p
    return ex


def _synth_odds(orace, k):
    """三連単オッズ→合成二車単オッズ {(a,b): odds}。k=控除校正係数。"""
    q = implied_trifecta_probs(orace)
    qe = defaultdict(float)
    for (a, b, c), v in q.items():
        qe[(a, b)] += v
    return {ab: (k / v) for ab, v in qe.items() if v > 0}


# ── 選定ポリシー: (model_exacta, synth_odds) → 買い目 list[(a,b)] ──
def _rank_prob(mex):
    return [ab for ab, _ in sorted(mex.items(), key=lambda kv: -kv[1])]


def _fill(primary, backup, n):
    """primary→backup の順で重複を除きつつ n 点埋める。"""
    out = []
    for ab in list(primary) + list(backup):
        if ab and ab not in out:
            out.append(ab)
        if len(out) >= n:
            break
    return out[:n]


def policies(st, mex, so):
    ev = {ab: mex.get(ab, 0) * so.get(ab, 0) for ab in mex}
    rp = _rank_prob(mex)
    rev = [ab for ab, _ in sorted(ev.items(), key=lambda kv: -kv[1])]
    pool = [ab for ab in rp[:10]]
    ana = sorted(pool, key=lambda ab: -so.get(ab, 0))                    # プール内オッズ降順=穴
    cars = sorted(st, key=st.get, reverse=True)
    c1 = cars[0] if cars else None                                      # ◎
    c2 = cars[1] if len(cars) > 1 else None                            # ○
    av = [ab for ab in rp if ab[0] == c1][:4]                           # ◎1着相手top4(prob)
    av_ana = sorted(av, key=lambda ab: -so.get(ab, 0))                  # ◎1着で相手を穴寄り
    evpos = [ab for ab in rev if ev.get(ab, 0) >= 1.0]
    # ユーザー案: ◎1着でEVが無い場合に ◎2着 / ○軸 へ逃がす柔軟型
    f1 = [ab for ab in rp if ab[0] == c1]                              # ◎1着(prob順)
    f1_ev = [ab for ab in f1 if ev.get(ab, 0) >= 1.0]                  # ◎1着 EV≥1
    s1_ev = [ab for ab in sorted([ab for ab in rp if ab[1] == c1],
                                 key=lambda ab: -ev.get(ab, 0)) if ev.get(ab, 0) >= 1.0]  # ◎2着 EV≥1
    f2_ev = [ab for ab in sorted([ab for ab in rp if ab[0] == c2],
                                 key=lambda ab: -ev.get(ab, 0)) if ev.get(ab, 0) >= 1.0]  # ○1着 EV≥1
    av_ana_ev = [ab for ab in av_ana if ev.get(ab, 0) >= 1.0]
    s1_ana = sorted([ab for ab in rp if ab[1] == c1], key=lambda ab: -so.get(ab, 0))  # ◎2着 人気薄
    return {
        "P2 本命prob3": rp[:3],
        "P8 ◎軸-相手穴3": av_ana[:3],
        # --- 柔軟型(◎1着でEV無→◎2着/○軸へ) ---
        "FA ◎1着EV≥1優先→prob3": _fill(sorted(f1_ev, key=lambda ab: -ev[ab]), f1, 3),
        "FB ◎1着EV→◎2着EV→prob3": _fill(f1_ev + s1_ev, f1, 3),
        "FC ◎/○1着EV≥1 prob3": _fill(sorted(f1_ev + f2_ev, key=lambda ab: -mex[ab]), f1, 3),
        "FD ◎1着EV→◎2着EV→○1着EV→prob": _fill(f1_ev + s1_ev + f2_ev, f1, 3),
        "FE ◎軸穴EV→無ければ◎2着穴": _fill(av_ana_ev + s1_ana, av_ana, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()
    model, _, _ = load_for(False)
    base = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    rids = {s.race_id for s in samples}
    exa, odds, lines = _load(args.db, rids)
    samples = [s for s in samples if s.race_id in exa and s.race_id in odds and s.race_id in lines]
    samples.sort(key=lambda s: (s.date, s.race_id))
    n = len(samples)
    bounds = fold_boundaries(n, n_folds=args.folds, warmup_frac=0.40, window="expanding")

    # 控除校正係数 k: 全レースの勝ち払戻平均 = 合成オッズ(k=1)の勝ち組平均 になる k を1回で推定
    num = den = 0.0
    for s in samples:
        q = implied_trifecta_probs(odds[s.race_id])
        qe = defaultdict(float)
        for (a, b, c), v in q.items():
            qe[(a, b)] += v
        w = exa[s.race_id][0]
        if tuple(w) in qe and qe[tuple(w)] > 0:
            num += exa[s.race_id][1] / 100.0
            den += 1.0 / qe[tuple(w)]
    k = num / den if den else 0.75
    print(f"男子7車 {samples[0].date}〜{samples[-1].date}  {n}レース  as-of walk-forward "
          f"{len(bounds)}fold  合成二車単オッズ控除校正 k={k:.3f}\n")

    names = list(policies({1: 0.6, 2: 0.4}, {(1, 2): 1.0}, {(1, 2): 2.0}).keys())
    # per policy per fold: [pts, ret, hit, oddsum(of hits)]
    stat = {nm: [defaultdict(float) for _ in bounds] for nm in names}
    for fi, (a, b, c) in enumerate(bounds):
        m = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            st = m.strengths(s.X, s.car_numbers)
            if len(st) < 4:
                continue
            mex = _model_exacta(st, lines[s.race_id])
            so = _synth_odds(odds[s.race_id], k)
            combo, pay = exa[s.race_id]; combo = tuple(combo)
            for nm, picks in policies(st, mex, so).items():
                d = stat[nm][fi]
                d["pts"] += len(picks)
                if combo in picks:
                    d["hit"] += 1; d["ret"] += pay
                d["n"] += 1

    def froi(d):
        return d["ret"] / (d["pts"] * 100) * 100 if d["pts"] else 0
    print(f"{'ポリシー':<26}{'平均点/R':>8}{'的中率':>8}{'統合ROI':>9}{'fold毎ROI(平均±SD)':>20}{'最低fold':>9}")
    rows = []
    for nm in names:
        folds = stat[nm]
        rois = [froi(d) for d in folds if d["pts"] > 0]
        tot = defaultdict(float)
        for d in folds:
            for k2 in ("pts", "ret", "hit", "n"):
                tot[k2] += d[k2]
        roi = froi(tot)
        hr = tot["hit"] / tot["n"] * 100 if tot["n"] else 0
        ppr = tot["pts"] / tot["n"] if tot["n"] else 0
        sd = float(np.std(rois)) if rois else 0
        rows.append((nm, ppr, hr, roi, float(np.mean(rois)) if rois else 0, sd, min(rois) if rois else 0))
    for nm, ppr, hr, roi, mroi, sd, mn in rows:
        print(f"{nm:<26}{ppr:>7.1f}{hr:>7.1f}%{roi:>8.1f}%{mroi:>10.1f}±{sd:>4.1f}{mn:>8.1f}%")
    print("\n※ 安定性=fold毎ROIのSDが小さい。回収率=統合ROI。両立(高ROI×低SD)を選ぶ。")


if __name__ == "__main__":
    main()
