"""Phase4 バケット分析の実行（本戦略の核）。

学習期間でPL線形を訓練し、検証期間（未来）で「レースタイプ×オッズ帯」バケット別の実現ROIと
キャリブレーションを集計、favorite-longshot bias を確認、EV閾値をチューニングして狙い目マップを出す。

  python scripts/run_bucket_analysis.py --db data/keirin.sqlite

★狙い目マップができるまで実弾投入しない（仕様書）。これは検証であって黒字保証ではない。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, EV_THRESHOLDS, EV_GUARD
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.model.evaluate import time_split
from src.backtest.bucket_analysis import build_records, bucket_roi, odds_bucket_roi

MIN_BETS = 30   # これ未満の買い目数のバケットはサンプル不足として参考扱い


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase4 バケット分析")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--haircut", type=float, default=1.0, help="オッズhaircut係数(暫定1.0)")
    args = ap.parse_args()

    # 本番と同じ31特徴(拡張20+rel_elo+展開10)を as-of 付与し、train期間のみで lambdarank を学習
    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    samples = augment_samples(base, args.db, feats31)
    train, test = time_split(samples, args.test_frac)
    print(f"サンプル {len(samples)}（train {len(train)} / test {len(test)}） 特徴{len(samples[0].feature_names)}列")
    print(f"検証期間: {test[0].date} 〜 {test[-1].date}\n")

    model = train_gbdt(train)                        # 検証期間は out-of-sample（リーク無し）
    test_ids = [s.race_id for s in test]
    records = build_records(args.db, model, test_ids, haircut=args.haircut)
    print(f"検証買い目レコード: {len(records)}（{len(test_ids)}レース × 各210点）\n")

    # --- favorite-longshot bias（オッズ帯別・全買い目の実現ROI） ---
    print("=== favorite-longshot 確認（オッズ帯別・全210点機械買いのROI） ===")
    print(f"{'オッズ帯':>10}{'点数':>8}{'的中':>6}{'ROI':>9}")
    for bucket, v in odds_bucket_roi(records).items():
        roi = f"{v['roi']*100:.1f}%" if v['roi'] is not None else "-"
        print(f"{bucket:>10}{v['n']:>8}{v['hits']:>6}{roi:>9}")

    # --- バケット別ROI＋較正（EV閾値チューニング） ---
    guard_min, guard_max = EV_GUARD["min_prob"], EV_GUARD["max_odds"]
    for thr in EV_THRESHOLDS:
        print(f"\n=== バケット別 実現ROI（EV閾値 {thr} / min_prob {guard_min} / max_odds {guard_max}） ===")
        res = bucket_roi(records, ev_threshold=thr, min_prob=guard_min, max_odds=guard_max)
        print(f"{'レースタイプ':>8}{'オッズ帯':>10}{'買い目':>7}{'的中':>5}{'ROI':>9}{'ECE':>8}")
        # ROI降順で表示
        for key, v in sorted(res.items(), key=lambda kv: (kv[1]['roi'] is None, -(kv[1]['roi'] or 0))):
            if v["n_bets"] == 0:
                continue
            roi = f"{v['roi']*100:.1f}%" if v['roi'] is not None else "-"
            ece = f"{v['ece']:.3f}" if v['ece'] is not None else "-"
            print(f"{key[0]:>8}{key[1]:>10}{v['n_bets']:>7}{v['n_hits']:>5}{roi:>9}{ece:>8}")

    # --- 狙い目マップ（ROI>100% かつ サンプル十分 かつ 較正良好） ---
    print(f"\n=== 狙い目マップ候補（EV閾値横断: ROI>100% & 買い目>={MIN_BETS} & ECE<0.05） ===")
    found = False
    for thr in EV_THRESHOLDS:
        res = bucket_roi(records, ev_threshold=thr, min_prob=guard_min, max_odds=guard_max)
        for key, v in res.items():
            if (v["roi"] and v["roi"] > 1.0 and v["n_bets"] >= MIN_BETS
                    and v["ece"] is not None and v["ece"] < 0.05):
                found = True
                print(f"  {key[0]} × {key[1]} @EV>={thr}: ROI {v['roi']*100:.1f}% "
                      f"（買い目{v['n_bets']}・的中{v['n_hits']}・ECE {v['ece']:.3f}）")
    if not found:
        print("  該当なし。サンプル増加・特徴量拡充(recent_form)・haircut推定後に再検証する。")
        print("  （現状は競走得点中心の限定特徴量。favorite-longshot と控除率でROIは1.0未満が基本）")


if __name__ == "__main__":
    main()
