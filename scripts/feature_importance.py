"""特徴量重要度分析（C）。3年+Eloで PL線形 と LightGBM を学習し、重要度を比較する。

  python scripts/feature_importance.py --db data/keirin.sqlite

出力:
  - PL線形の標準化重み（符号つき・|w|降順）
  - LightGBMのgain重要度（%・降順）
  - （shapが入っていれば）平均|SHAP|
  - 参考: PL vs GBDT の時系列検証指標（top1/logloss/brier/ece）
Elo(rel_elo)を含めた21特徴で評価し、Tier妥当性・冗長特徴の見直しに使う。
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
from src.model.train_pl import train_pl
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate, time_split
from src.model.elo import compute_pre_race_elo, DEFAULT_ELO


def _augment(samples, pre_elo):
    out = []
    for s in samples:
        s2 = copy.copy(s)
        elos = np.array([pre_elo.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
        s2.X = np.hstack([s.X, (elos - elos.mean()).reshape(-1, 1)])
        s2.feature_names = list(s.feature_names) + ["rel_elo"]
        out.append(s2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="特徴量重要度分析（PL vs LightGBM, Elo込み）")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    pre_elo = compute_pre_race_elo(args.db)
    samples = _augment(base, pre_elo)
    feats = samples[0].feature_names
    print(f"サンプル {len(samples)}レース / 特徴量 {len(feats)}（Elo込み）\n")

    tr, te = time_split(samples, 0.25)
    pl = train_pl(tr)
    gb = train_gbdt(tr)

    # --- PL線形の重み ---
    print("=== PL線形 標準化重み（|w|降順） ===")
    for name, w in sorted(zip(pl.feature_names, pl.weights), key=lambda x: -abs(x[1])):
        print(f"  {name:<20}{w:+.4f}  {'#'*int(min(abs(w),1.5)*20)}")

    # --- LightGBM gain重要度 ---
    gain = gb.booster.feature_importance(importance_type="gain").astype(float)
    tot = gain.sum() or 1.0
    print("\n=== LightGBM gain重要度（%降順） ===")
    for name, g in sorted(zip(feats, gain), key=lambda x: -x[1]):
        pct = 100 * g / tot
        print(f"  {name:<20}{pct:>6.1f}%  {'#'*int(pct/2)}")

    # --- SHAP（あれば） ---
    try:
        import shap
        allX = np.vstack([(s.X - gb.mean) / gb.std if gb.standardize_x else s.X
                          for s in te[:400]])
        expl = shap.TreeExplainer(gb.booster)
        sv = np.abs(expl.shap_values(allX)).mean(axis=0)
        print("\n=== 平均|SHAP|（降順, GBDT・検証データ） ===")
        for name, v in sorted(zip(feats, sv), key=lambda x: -x[1]):
            print(f"  {name:<20}{v:.4f}")
    except Exception as e:
        print(f"\n（SHAPは未実行: {type(e).__name__}。gain重要度で代替。pip install shap で有効化）")

    # --- 参考指標 ---
    rp, rg = evaluate(pl.strengths, te), evaluate(gb.strengths, te)
    print(f"\n=== 参考: 時系列検証（PL線形 vs LightGBM, Elo込み） ===")
    print(f"{'指標':<10}{'PL線形':>12}{'LightGBM':>12}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{k:<10}{rp[k]:>12}{rg[k]:>12}")


if __name__ == "__main__":
    main()
