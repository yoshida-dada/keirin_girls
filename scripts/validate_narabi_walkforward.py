"""並び予想特徴の「混戦での効き」を walk-forward 多foldで検証（全6414レースにnarabiあり）。

単一分割では 31特徴 vs +並び予想 で混戦 三連単top10 が +7.4pt(n=68)だったが小サンプル。複数の
時系列foldで、混戦サブセットの top1/三連単top10 が安定して上振れするかを確認する。全体も併記。

  PYTHONIOENCODING=utf-8 python scripts/validate_narabi_walkforward.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate
from src.model.feature_augment import augment_samples
from src.model.race_type import classify_race
from src.model.plackett_luce import all_trifecta_probs
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import compute_narabi_features, NARABI_KEYS
from src.backtest.walkforward import fold_boundaries


def _tri10(model, test):
    if not test:
        return 0.0
    hit = 0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        ranked = [k for k, _ in sorted(all_trifecta_probs(st).items(), key=lambda kv: -kv[1])]
        hit += int(tuple(s.order[:3]) in ranked[:10])
    return hit / len(test)


def _rel(vals):
    present = [v for v in vals if v is not None]
    mean = sum(present) / len(present) if present else 0.0
    return [(v - mean) if v is not None else 0.0 for v in vals]


def main():
    ap = argparse.ArgumentParser(description="並び予想の混戦効きを walk-forward 検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    s31 = augment_samples(base, args.db, feats31)
    narabi = compute_narabi_features(args.db)

    def add_narabi(samples):
        out = []
        for s in samples:
            s2 = copy.copy(s)
            cols = []
            for key in NARABI_KEYS:
                vals = [narabi.get((s.race_id, c), {}).get(key) for c in s.car_numbers]
                col = vals if key == "narabi_lead" else _rel(vals)
                cols.append(np.array([v if v is not None else 0.0 for v in col]).reshape(-1, 1))
            s2.X = np.hstack([s.X] + cols)
            s2.feature_names = list(s.feature_names) + NARABI_KEYS
            out.append(s2)
        return out

    s_nb = add_narabi(s31)     # s31 と並列（同一index=同一レース）
    bounds = fold_boundaries(len(s31), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"walk-forward {len(bounds)}fold（expanding, warmup40%）／全レースにnarabiあり\n")
    print(f"{'fold':>4}{'検証期間':>24}{'混戦R':>6}"
          f"{'  混戦tri10 31→+並び':>22}{'  混戦top1 31→+並び':>20}{' 全体tri10 31→+並び':>20}")

    agg = []
    for i, (a, b, c) in enumerate(bounds):
        m31 = train_gbdt(s31[a:b]); mnb = train_gbdt(s_nb[a:b])
        te31, tenb = s31[b:c], s_nb[b:c]
        lab = [classify_race(m31.strengths(s.X, s.car_numbers)).label for s in te31]
        ch31 = [te31[j] for j in range(len(te31)) if lab[j] == "混戦"]
        chnb = [tenb[j] for j in range(len(tenb)) if lab[j] == "混戦"]
        t31, tnb = _tri10(m31, ch31), _tri10(mnb, chnb)
        c31, cnb = evaluate(m31.strengths, ch31), evaluate(mnb.strengths, chnb)
        ot31, otnb = _tri10(m31, te31), _tri10(mnb, tenb)
        agg.append((t31, tnb, c31.get("top1_acc", 0), cnb.get("top1_acc", 0), ot31, otnb, len(ch31)))
        d0, d1 = te31[0].date, te31[-1].date
        print(f"{i:>4}{d0+'〜'+d1:>24}{len(ch31):>6}"
              f"{t31*100:>10.1f}→{tnb*100:.1f}%{c31.get('top1_acc',0)*100:>10.1f}→{cnb.get('top1_acc',0)*100:.1f}%"
              f"{ot31*100:>10.1f}→{otnb*100:.1f}%")

    n = len(agg)
    if n:
        wins_tri = sum(1 for r in agg if r[1] > r[0])
        wins_top1 = sum(1 for r in agg if r[3] > r[2])
        mean_d_tri = sum(r[1] - r[0] for r in agg) / n
        mean_d_top1 = sum(r[3] - r[2] for r in agg) / n
        print(f"\n混戦 三連単top10: +並びが勝ったfold {wins_tri}/{n} / 平均Δ {mean_d_tri*100:+.1f}pt")
        print(f"混戦 top1       : +並びが勝ったfold {wins_top1}/{n} / 平均Δ {mean_d_top1*100:+.1f}pt")
        print("判定: 大半のfoldで混戦tri10が+側かつ平均Δ>0 → 混戦での効きは再現性あり。"
              "符号がばらつく/平均≈0 → 単一分割の偶然。")


if __name__ == "__main__":
    main()
