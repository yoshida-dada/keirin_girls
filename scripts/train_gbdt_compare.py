"""PL線形 vs LightGBM(lambdarank) を収集済みデータで時系列比較する（S3）。

  python scripts/train_gbdt_compare.py --db data/keirin.sqlite

同一の time_split（前80%学習→後20%検証）で両モデルを学習し、evaluate() の
top1_acc / logloss / brier / ece を並べて表示する。recent_form/age 未取得のため
特徴量は限定的（racing_score/gear_ratio/rel_score_max/score_rank/脚質フラグ）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES
from src.model.train_pl import train_pl
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate, baseline_strengths, time_split


def main() -> None:
    ap = argparse.ArgumentParser(description="PL線形 vs LightGBM の時系列比較")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    samples = load_samples(args.db)
    print(f"学習サンプル（7車・結果あり）: {len(samples)}レース")
    print(f"特徴量: {PL_FEATURES}")
    train, test = time_split(samples, args.test_frac)
    print(f"時系列分割: train {len(train)} / test {len(test)}"
          f"（{train[0].date}〜{train[-1].date} → {test[0].date}〜{test[-1].date}）\n")

    base = baseline_strengths(PL_FEATURES)
    pl = train_pl(train, l2=args.l2)
    gbdt = train_gbdt(train, l2=args.l2)

    m_base = evaluate(base, test)
    m_pl = evaluate(pl.strengths, test)
    m_gbdt = evaluate(gbdt.strengths, test)

    print(f"{'指標':<10}{'ベースライン':>14}{'PL線形':>12}{'LightGBM':>14}")
    for key in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{key:<12}{m_base[key]:>12}{m_pl[key]:>12}{m_gbdt[key]:>14}")
    print("\n※ top1_acc↑ / logloss↓ / brier↓ / ece↓ が良い。ece=1着確率の較正誤差（課題B）。")
    print("※ recent_form/age 未取得のため特徴量限定。付与で更に改善余地あり。")


if __name__ == "__main__":
    main()
