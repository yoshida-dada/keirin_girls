"""Walk-forward 検証（Phase4の狙い目マップの安定性確認）。

単一分割のROIは高分散なので、時系列を複数フォールドに分けて「拡大窓（expanding window）」で
検証する。各フォールドで過去データだけを使って学習→未来ブロックで買いシミュレーション。
バケットごとのROIがフォールド間で安定して100%超かを見る（偶然か本物かの判別）。

  python scripts/walk_forward.py --db data/keirin.sqlite --folds 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, EV_GUARD
from src.model.training_data import load_samples, PL_FEATURES, PL_FEATURES_FULL
from src.model.train_pl import train_pl
from src.backtest.bucket_analysis import build_records, bucket_roi

EV_THRESHOLD = 1.10
MIN_BETS_PER_FOLD = 10       # フォールド内でこれ未満の買い目数は参考外
WARMUP_FRAC = 0.40           # 最初の学習に使う割合


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward バケット検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--ev", type=float, default=EV_THRESHOLD)
    ap.add_argument("--full", action="store_true", help="拡張特徴量(recent_form/age)を使う")
    args = ap.parse_args()

    features = PL_FEATURES_FULL if args.full else PL_FEATURES
    samples = load_samples(args.db, features=features)
    print(f"特徴量: {'拡張' if args.full else '基本'}({len(features)}個)")
    n = len(samples)
    warmup = int(n * WARMUP_FRAC)
    block = (n - warmup) // args.folds
    gmin, gmax = EV_GUARD["min_prob"], EV_GUARD["max_odds"]
    print(f"サンプル {n} / warmup {warmup} / fold {args.folds}（各 {block}レース）/ EV>={args.ev}\n")

    # per-bucket に各フォールドの (roi, n_bets, n_hits) を貯める
    fold_stats: list[dict] = []
    per_bucket: dict[tuple, list] = {}
    for i in range(args.folds):
        tr_end = warmup + i * block
        te_start, te_end = tr_end, (warmup + (i + 1) * block if i < args.folds - 1 else n)
        train, test = samples[:tr_end], samples[te_start:te_end]
        model = train_pl(train)
        recs = build_records(args.db, model, [s.race_id for s in test])
        res = bucket_roi(recs, ev_threshold=args.ev, min_prob=gmin, max_odds=gmax)
        fold_stats.append({"i": i, "period": f"{test[0].date}〜{test[-1].date}",
                           "n_train": len(train), "n_test": len(test)})
        print(f"[fold {i}] 学習{len(train)} → 検証{len(test)}（{test[0].date}〜{test[-1].date}）")
        for key, v in res.items():
            per_bucket.setdefault(key, []).append((v["roi"], v["n_bets"], v["n_hits"], v["ece"]))

    # 集計: フォールド間の安定性
    print(f"\n=== バケット別 walk-forward ROI（EV>={args.ev}） ===")
    print(f"{'レースタイプ':>8}{'オッズ帯':>10}{'総買い目':>8}{'総的中':>7}{'pooled ROI':>11}"
          f"{'>100%fold':>10}")
    rows = []
    for key, folds in per_bucket.items():
        valid = [(roi, nb, nh, ece) for roi, nb, nh, ece in folds if nb >= MIN_BETS_PER_FOLD]
        tot_bets = sum(nb for _, nb, _, _ in folds)
        tot_hits = sum(nh for _, _, nh, _ in folds)
        # pooled ROI = 総回収/総投票（各フォールドのstake=nb*100, ret=roi*stake）
        stake = sum(nb * 100 for _, nb, _, _ in folds)
        ret = sum((roi or 0) * nb * 100 for roi, nb, _, _ in folds)
        pooled = ret / stake if stake else None
        n_win_folds = sum(1 for roi, _, _, _ in valid if roi and roi > 1.0)
        rows.append((key, tot_bets, tot_hits, pooled, n_win_folds, len(valid)))
    for key, tb, th, pooled, nwf, nvalid in sorted(
            rows, key=lambda r: (r[3] is None, -(r[3] or 0))):
        if tb == 0:
            continue
        proi = f"{pooled*100:.1f}%" if pooled is not None else "-"
        print(f"{key[0]:>8}{key[1]:>10}{tb:>8}{th:>7}{proi:>11}{f'{nwf}/{nvalid}':>10}")

    print(f"\n※ pooled ROI=全フォールド通算の実現ROI。>100%fold=買い目十分なフォールドのうちROI>100%の数。")
    print("※ 全フォールドで安定して>100%なら本物に近い。1フォールドだけ突出はノイズ疑い（高分散）。")
    print("※ haircut未適用(1.0)・recent_form未取得の限定特徴量。実弾投入はさらなる検証後。")


if __name__ == "__main__":
    main()
