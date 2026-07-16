"""記事知見「中団(4番手)×捲りが勝つ・予想先頭(逃げ)は身を挺す役」を特徴化して検証。

現行34特徴(拡張20+Elo+展開10+並び3: narabi_pos/lead/leg)に、並び予想の隊列位置から
  narabi_mid   : 予想隊列が中団(位置index 2..4 = 3〜5番手)なら1（記事: 4番手最勝率）
  narabi_midleg: narabi_mid × 脚質前がかり度(自在=1等) ＝ 中団の自力型(捲り想定)
を足した36特徴を、walk-forward 5fold で比較する。混戦のtri10/top1と全体を見て、
現行34特徴を上回るか（記事の中団×捲りが並び予想の生位置を超える純増を生むか）を判定。

  PYTHONIOENCODING=utf-8 python scripts/validate_narabi_mid.py --db data/keirin.sqlite
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

MID_KEYS = ["narabi_mid", "narabi_midleg"]


def _tri10(model, test):
    if not test:
        return 0.0
    return sum(int(tuple(s.order[:3]) in
                   [k for k, _ in sorted(all_trifecta_probs(model.strengths(s.X, s.car_numbers)).items(),
                                         key=lambda kv: -kv[1])][:10]) for s in test) / len(test)


def _rel(vals):
    present = [v for v in vals if v is not None]
    m = sum(present) / len(present) if present else 0.0
    return [(v - m) if v is not None else 0.0 for v in vals]


def main():
    ap = argparse.ArgumentParser(description="中団×捲り特徴の検証(34 vs 36)")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    s31 = augment_samples(base, args.db, list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES))
    narabi = compute_narabi_features(args.db)

    def add_narabi(samples, with_mid):
        out = []
        for s in samples:
            s2 = copy.copy(s)
            cars = list(s.car_numbers)
            # 並び3列（pos/leg 相対化, lead 生）
            pos_rel = _rel([narabi.get((s.race_id, c), {}).get("narabi_pos") for c in cars])
            leg_rel = _rel([narabi.get((s.race_id, c), {}).get("narabi_leg") for c in cars])
            lead = [narabi.get((s.race_id, c), {}).get("narabi_lead") or 0.0 for c in cars]
            cols = [pos_rel, lead, leg_rel]
            fn = NARABI_KEYS
            if with_mid:
                mid, midleg = [], []
                for c in cars:
                    d = narabi.get((s.race_id, c), {})
                    p = d.get("narabi_pos")
                    m = 1.0 if (p is not None and 2 <= p <= 4) else 0.0   # 中団(3〜5番手)
                    mid.append(m)
                    midleg.append(m * (d.get("narabi_leg") or 0.0))       # 中団×前がかり度
                cols += [mid, midleg]
                fn = NARABI_KEYS + MID_KEYS
            mat = np.array(cols, dtype=float).T                          # (ncar, ncol)
            s2.X = np.hstack([s.X, mat])
            s2.feature_names = list(s.feature_names) + list(fn)
            out.append(s2)
        return out

    s34 = add_narabi(s31, with_mid=False)
    s36 = add_narabi(s31, with_mid=True)
    bounds = fold_boundaries(len(s31), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"walk-forward {len(bounds)}fold（34=並び3 vs 36=+中団2）\n")
    print(f"{'fold':>4}{'混戦R':>6}{'  混戦tri10 34→36':>20}{'  混戦top1 34→36':>18}{' 全体tri10 34→36':>18}{' 全体ece 34→36':>18}")
    agg = []
    for i, (a, b, c) in enumerate(bounds):
        m34, m36 = train_gbdt(s34[a:b]), train_gbdt(s36[a:b])
        te34, te36 = s34[b:c], s36[b:c]
        lab = [classify_race(m34.strengths(s.X, s.car_numbers)).label for s in te34]
        ch34 = [te34[j] for j in range(len(te34)) if lab[j] == "混戦"]
        ch36 = [te36[j] for j in range(len(te36)) if lab[j] == "混戦"]
        t34, t36 = _tri10(m34, ch34), _tri10(m36, ch36)
        c34, c36 = evaluate(m34.strengths, ch34), evaluate(m36.strengths, ch36)
        o34, o36 = evaluate(m34.strengths, te34), evaluate(m36.strengths, te36)
        ot34, ot36 = _tri10(m34, te34), _tri10(m36, te36)
        agg.append((t36 - t34, c36.get("top1_acc", 0) - c34.get("top1_acc", 0),
                    ot36 - ot34, o36["ece"] - o34["ece"]))
        print(f"{i:>4}{len(ch34):>6}{t34*100:>9.1f}→{t36*100:.1f}%{c34.get('top1_acc',0)*100:>9.1f}→{c36.get('top1_acc',0)*100:.1f}%"
              f"{ot34*100:>9.1f}→{ot36*100:.1f}%{o34['ece']:>10.4f}→{o36['ece']:.4f}")
    n = len(agg)
    if n:
        wtri = sum(1 for r in agg if r[0] > 0)
        print(f"\n中団2列の効果(36−34): 混戦tri10 +勝ち{wtri}/{n}・平均Δ{sum(r[0] for r in agg)/n*100:+.1f}pt / "
              f"混戦top1 平均Δ{sum(r[1] for r in agg)/n*100:+.1f}pt")
        print(f"  全体tri10 平均Δ{sum(r[2] for r in agg)/n*100:+.2f}pt / 全体ece 平均Δ{sum(r[3] for r in agg)/n:+.5f}")
        print("判定: 混戦tri10が大半のfoldで+かつ全体ece悪化なし → 中団特徴を採用(36特徴)。")


if __name__ == "__main__":
    main()
