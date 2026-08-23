"""出走間隔 E1(線形min120/30) の採否決定: top1/tri10 に加え ECE/logloss/brier を確認(男子)。

過去の選手ローリング(線形中何日)は top1微改善だが ECE悪化 で非採用だった。E1 も同じ轍か、
較正を崩さず純増するかを walk-forward で確認する。
  PYTHONIOENCODING=utf-8 python scripts/validate_interval_e1.py --folds 5
"""
from __future__ import annotations

import argparse
import copy
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate
from src.model.plackett_luce import all_trifecta_probs
from src.backtest.walkforward import fold_boundaries
from src.features.interval_features import compute_pre_race_gap

DB = str(DATA_DIR / "keirin_men.sqlite")


def _tri10(model, test):
    if not test:
        return 0.0
    hit = 0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        ranked = [k for k, _ in sorted(all_trifecta_probs(st).items(), key=lambda kv: -kv[1])]
        hit += int(tuple(s.order[:3]) in ranked[:10])
    return hit / len(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    model, _, _ = load_for(False)
    base = load_samples(DB, field_size=[7, 9], features=PL_FEATURES_FULL)
    s0 = augment_samples(base, DB, model.feature_names)
    gap = compute_pre_race_gap(DB)          # 本番と同一(results由来, keyed by (rid,car))

    def gap_of(s, car):
        return gap.get((s.race_id, car))

    def add(samples):
        out = []
        for s in samples:
            col = np.array([[(min(gap_of(s, c), 120) / 30.0) if gap_of(s, c) else 0.0]
                            for c in s.car_numbers])
            s2 = copy.copy(s)
            s2.X = np.hstack([s.X, col])
            s2.feature_names = list(s.feature_names) + ["gap_lin"]
            out.append(s2)
        return out

    sg = add(s0)
    bounds = fold_boundaries(len(s0), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"男子 walk-forward {len(bounds)}fold  E1(線形gap) 採否判定\n")
    print(f"{'fold':>4}{'top1 base→E1':>18}{'tri10 base→E1':>18}{'ece base→E1':>20}{'logloss base→E1':>22}")
    agg = defaultdict(list)
    for i, (a, b, c) in enumerate(bounds):
        m0, mg = train_gbdt(s0[a:b]), train_gbdt(sg[a:b])
        te0, teg = s0[b:c], sg[b:c]
        c0, cg = evaluate(m0.strengths, te0), evaluate(mg.strengths, teg)
        t0, tg = _tri10(m0, te0), _tri10(mg, teg)
        agg["t1"].append((c0["top1_acc"], cg["top1_acc"]))
        agg["t10"].append((t0, tg))
        agg["ece"].append((c0["ece"], cg["ece"]))
        agg["ll"].append((c0["logloss"], cg["logloss"]))
        agg["br"].append((c0["brier"], cg["brier"]))
        print(f"{i:>4}{c0['top1_acc']*100:>10.1f}→{cg['top1_acc']*100:.1f}%"
              f"{t0*100:>11.1f}→{tg*100:.1f}%"
              f"{c0['ece']:>13.4f}→{cg['ece']:.4f}{c0['logloss']:>14.4f}→{cg['logloss']:.4f}")
    n = len(bounds)
    def stat(k, better_low=False):
        ds = [b - a for a, b in agg[k]]
        wins = sum((d < 0) if better_low else (d > 0) for d in ds)
        return sum(ds) / n, wins
    for k, lbl, low in [("t1", "top1", False), ("t10", "tri10", False),
                        ("ece", "ECE", True), ("ll", "logloss", True), ("br", "brier", True)]:
        m, w = stat(k, low)
        arrow = "改善" if (m < 0) == low else "悪化"
        unit = "pt" if k in ("t1", "t10") else ""
        val = m * 100 if k in ("t1", "t10") else m
        print(f"  {lbl:<8} 平均Δ {val:+.4f}{unit}  E1優位fold {w}/{n}  ({arrow})")
    print("\n判定: top1が5/5で+, かつ ECE/logloss/brier が悪化しない → 採用可。ECE悪化なら過去のローリングと同轍。")


if __name__ == "__main__":
    main()
