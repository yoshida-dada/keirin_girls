"""予測実績の自動照合（D13）。検証期間(out-of-sample)でモデル予測を実結果と突き合わせる。

  python scripts/verify_predictions.py --db data/keirin.sqlite

指標:
  - 1着的中率   : モデルの1着本命が実際に1着だった割合
  - 3着内的中率 : 実際の1着がモデル上位3車に入っていた割合
  - 三連単 上位N的中: モデル三連単上位Nに実着順が含まれた割合（N=1/5/10/30）
  - Brier(1着)  : 1着確率の較正
build_accuracy_section(db) は dashboard/data.json 用の dict を返す（build_predictionsから利用）。
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
from src.model.persist import load_model
from src.model.evaluate import time_split
from src.model.elo import compute_pre_race_elo, DEFAULT_ELO
from src.model.plackett_luce import all_trifecta_probs
from src.backtest.calibration import brier_score


def _augment_if_elo(db_path, model, samples):
    """モデルの feature_names に合わせて rel_elo / 展開特徴 を付与（共通関数・skew防止）。"""
    from src.model.feature_augment import augment_samples
    return augment_samples(samples, db_path, model.feature_names)


def build_accuracy_section(db_path, test_frac: float = 0.25) -> dict:
    """検証期間でモデルの的中実績を集計して dict を返す（dashboard用）。"""
    model = load_model()
    samples = load_samples(db_path, features=PL_FEATURES_FULL)
    samples = _augment_if_elo(db_path, model, samples)
    _, test = time_split(samples, test_frac)
    n = top1 = top3 = tri1 = tri5 = tri10 = tri30 = 0
    pairs = []
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        if not st or len(s.order) < 3:
            continue
        winner = s.order[0]
        ranked = sorted(st, key=lambda c: -st[c])
        top1 += int(ranked[0] == winner)
        top3 += int(winner in ranked[:3])
        probs = all_trifecta_probs(st)
        tri_rank = [c for c, _ in sorted(probs.items(), key=lambda kv: -kv[1])]
        actual = tuple(s.order[:3])
        pos = tri_rank.index(actual) + 1 if actual in tri_rank else 9999
        tri1 += int(pos <= 1); tri5 += int(pos <= 5)
        tri10 += int(pos <= 10); tri30 += int(pos <= 30)
        for c, p in st.items():
            pairs.append((p, 1 if c == winner else 0))
        n += 1
    if n == 0:
        return {"status": "pending", "note": "検証データ不足"}
    return {
        "status": "ok",
        "note": "検証期間（学習に使っていない直近区間）でのモデル予測の的中実績。",
        "period": f"{test[0].date}〜{test[-1].date}",
        "n_races": n,
        "top1_rate": round(top1 / n, 4),
        "top3_rate": round(top3 / n, 4),
        "trifecta": [
            {"topn": 1, "rate": round(tri1 / n, 4)},
            {"topn": 5, "rate": round(tri5 / n, 4)},
            {"topn": 10, "rate": round(tri10 / n, 4)},
            {"topn": 30, "rate": round(tri30 / n, 4)},
        ],
        "brier": round(brier_score(pairs), 5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="予測実績の照合（D13）")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    args = ap.parse_args()
    a = build_accuracy_section(args.db)
    if a["status"] != "ok":
        print(a); return
    print(f"検証期間 {a['period']} / {a['n_races']}レース")
    print(f"  1着的中率  : {a['top1_rate']*100:.1f}%")
    print(f"  3着内的中率: {a['top3_rate']*100:.1f}%（実1着がモデル上位3車）")
    for t in a["trifecta"]:
        print(f"  三連単 上位{t['topn']:>2}的中: {t['rate']*100:.1f}%")
    print(f"  Brier(1着) : {a['brier']}")


if __name__ == "__main__":
    main()
