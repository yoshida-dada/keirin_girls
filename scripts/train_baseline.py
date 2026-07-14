"""PL線形モデルを収集済みデータで学習し、ベースラインと時系列比較する（S3）。

  python scripts/train_baseline.py --db data/keirin.sqlite

時系列分割（前80%学習→後20%検証）で、競走得点のみのベースラインに対する改善を確認する。
学習器の配線検証が目的で、recent_form未取得のため特徴量は限定的（docs/design_s2_features.md）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES, PL_FEATURES_FULL
from src.model.train_pl import train_pl
from src.model.evaluate import evaluate, baseline_strengths, time_split


def main() -> None:
    ap = argparse.ArgumentParser(description="PL線形モデルの学習と時系列評価")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--full", action="store_true", help="拡張特徴量(recent_form/age)を使う")
    args = ap.parse_args()

    features = PL_FEATURES_FULL if args.full else PL_FEATURES
    samples = load_samples(args.db, features=features)
    print(f"学習サンプル（7車・結果あり）: {len(samples)}レース")
    print(f"特徴量({'拡張' if args.full else '基本'}, {len(features)}個): {features}")
    train, test = time_split(samples, args.test_frac)
    print(f"時系列分割: train {len(train)} / test {len(test)}"
          f"（{train[0].date}〜{train[-1].date} → {test[0].date}〜{test[-1].date}）\n")

    model = train_pl(train, l2=args.l2)
    print("学習済み重み（標準化特徴量に対する係数）:")
    for name, w in zip(model.feature_names, model.weights):
        print(f"  {name:<18}{w:+.4f}")

    base = baseline_strengths(features)
    m_base = evaluate(base, test)
    m_pl = evaluate(model.strengths, test)
    print(f"\n{'指標':<12}{'ベースライン':>14}{'PL線形':>12}")
    for key in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{key:<12}{m_base[key]:>14}{m_pl[key]:>12}")
    print("\n※ top1_acc↑ / logloss↓ / brier↓ / ece↓ が良い。ece=1着確率の較正誤差（課題B）。")
    print("※ recent_form/age 未取得のため特徴量限定。付与で更に改善余地あり。")


if __name__ == "__main__":
    main()
