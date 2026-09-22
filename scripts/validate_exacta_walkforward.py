"""二車単 ◎→番手(弱い本命) エッジの as-of walk-forward 確定検証(男子7車)。

in-sampleリーク排除: 各foldで「そのfoldのtestより過去だけ」で lambdarank を再学習し、
そのas-ofモデルで◎勝率帯を判定→◎→番手を1点、実払戻(payouts_exacta)で決済。
  ・fold別ROI（直近foldでも100%超を維持するか＝時期減衰の排除）
  ・全fold統合(out-of-sample)のROIと、レース単位ブートストラップ95%区間（下限>100%か）
cap=0.30/0.34 の2閾値。判定基準: 統合区間の下限>100% かつ 過半foldでROI>100% かつ最新foldが極端に割れない。

  PYTHONIOENCODING=utf-8 python scripts/validate_exacta_walkforward.py --folds 6
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
from src.backtest.walkforward import fold_boundaries


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    exa = {rid: (tuple(int(x) for x in combo.split("-")), p)
           for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_exacta")}
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    c.close()
    mate = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        for ln in [x for x in ls if x]:
            for k in range(len(ln) - 1):
                mate[(rid, ln[k])] = ln[k + 1]
    return exa, mate


def _roi(bets):
    if not bets:
        return 0.0, 0.0, 0
    ret = sum(p for h, p in bets if h)
    hr = sum(1 for h, _ in bets if h) / len(bets) * 100
    return ret / (len(bets) * 100) * 100, hr, len(bets)


def _boot_ci(bets, B=3000, seed=0):
    if len(bets) < 20:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    pays = np.array([p if h else 0 for h, p in bets], dtype=float)
    n = len(pays)
    idx = rng.integers(0, n, size=(B, n))
    rois = pays[idx].sum(axis=1) / (n * 100) * 100
    return float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()
    exa, mate = _load(args.db)
    model, _, _ = load_for(False)          # 特徴セット(feature_names)を借りるだけ、再学習する

    base = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    samples = [s for s in samples if s.race_id in exa]
    samples.sort(key=lambda s: (s.date, s.race_id))
    n = len(samples)
    bounds = fold_boundaries(n, n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"男子7車 {samples[0].date}〜{samples[-1].date}  {n}レース  "
          f"as-of walk-forward {len(bounds)}fold(expanding, warmup40%)\n")

    # 各foldのas-ofモデルを1回だけ学習し、cap非依存の (band_pw, hit, pay) を貯める
    fold_bets = []   # fold_bets[i] = list of (pw, hit, pay)
    for a, b, c in bounds:
        m = train_gbdt(samples[a:b])
        rows = []
        for s in samples[b:c]:
            st = m.strengths(s.X, s.car_numbers)
            anchor = max(st, key=st.get)
            nb = mate.get((s.race_id, anchor))
            if not nb:
                continue
            pw = st[anchor] / sum(st.values())
            combo, pay = exa[s.race_id]
            rows.append((pw, tuple(combo) == (anchor, nb), pay))
        fold_bets.append(rows)

    for cap in (0.30, 0.34):
        pooled, fold_rois = [], []
        print(f"══ ◎勝率<{cap:.2f} × ◎→番手（as-of） ══")
        print(f"   {'fold':>4}{'検証期間':>24}{'点数':>7}{'的中率':>9}{'回収率':>9}")
        for i, (a, b, c) in enumerate(bounds):
            fb = [(h, p) for pw, h, p in fold_bets[i] if pw < cap]
            roi, hr, m_ = _roi(fb)
            fold_rois.append(roi)
            pooled += fb
            print(f"   {i:>4}{samples[b].date+'〜'+samples[c-1].date:>24}"
                  f"{m_:>7}{hr:>8.1f}%{roi:>8.1f}%{'  ★' if roi >= 100 else ''}")
        roi, hr, m_ = _roi(pooled)
        lo, hi = _boot_ci(pooled)
        wins = sum(1 for r in fold_rois if r >= 100)
        print(f"   {'統合':>4}{'(out-of-sample)':>24}{m_:>7}{hr:>8.1f}%{roi:>8.1f}%")
        print(f"        100%超fold {wins}/{len(fold_rois)}  最新fold {fold_rois[-1]:.1f}%")
        print(f"        ブートストラップ95%区間 ROI [{lo:.1f}% 〜 {hi:.1f}%]  下限>100%={'YES' if lo > 100 else 'NO'}\n")


if __name__ == "__main__":
    main()
