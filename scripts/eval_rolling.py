"""選手as-oフローリング特徴量の有用性検証（P3）。

現行本番特徴（拡張20 + rel_elo）に、rider_rolling の履歴特徴（通算勝率/直近5走平均着順/
バンク別勝率/中何日）を**レース内相対**にして足し、PL線形で改善するか比較する。

  python scripts/eval_rolling.py --db data/keirin.sqlite

リークなし: rider_rolling.compute_rolling は各エントリの発走前(as-of)値を返す。
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
from src.model.evaluate import evaluate, time_split
from src.model.elo import compute_pre_race_elo, DEFAULT_ELO
from src.features.rider_rolling import compute_rolling

# 追加するローリング特徴（レース内相対にする列名）
ROLL_FEATURES = ["r_career_win", "r_recent5_finish", "r_venue_win", "r_days_since"]
_RAW_KEYS = ["career_win_rate", "recent5_avg_finish", "venue_win_rate", "days_since_last"]


def _augment_elo(samples, pre_elo):
    out = []
    for s in samples:
        s2 = copy.copy(s)
        elos = np.array([pre_elo.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
        s2.X = np.hstack([s.X, (elos - elos.mean()).reshape(-1, 1)])
        s2.feature_names = list(s.feature_names) + ["rel_elo"]
        out.append(s2)
    return out


def _rel_column(vals: list[float | None]) -> np.ndarray:
    """レース内相対化: 有効値の平均を引く。欠損は0（=レース平均扱い）。"""
    present = [v for v in vals if v is not None]
    mean = sum(present) / len(present) if present else 0.0
    return np.array([(v - mean) if v is not None else 0.0 for v in vals])


def _augment_rolling(samples, rolling):
    out = []
    for s in samples:
        s2 = copy.copy(s)
        cols = []
        for key in _RAW_KEYS:
            vals = [rolling.get((s.race_id, c), {}).get(key) for c in s.car_numbers]
            cols.append(_rel_column(vals).reshape(-1, 1))
        s2.X = np.hstack([s.X] + cols)
        s2.feature_names = list(s.feature_names) + ROLL_FEATURES
        out.append(s2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ローリング特徴の有用性検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    print(f"サンプル {len(base)}レース")
    cur = _augment_elo(base, compute_pre_race_elo(args.db))          # 現行: 拡張+Elo
    withroll = _augment_rolling(cur, compute_rolling(args.db))        # +ローリング

    tr_c, te_c = time_split(cur, 0.25)
    tr_r, te_r = time_split(withroll, 0.25)
    m_c, m_r = train_pl(tr_c), train_pl(tr_r)
    r_c, r_r = evaluate(m_c.strengths, te_c), evaluate(m_r.strengths, te_r)

    print(f"\n{'指標':<10}{'現行(+Elo)':>13}{'+ローリング':>13}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{k:<10}{r_c[k]:>13}{r_r[k]:>13}")

    w = dict(zip(m_r.feature_names, m_r.weights))
    print("\nローリング特徴の学習重み:")
    for f in ROLL_FEATURES:
        print(f"  {f:<18}{w.get(f, 0):+.4f}")

    improved = (r_r["ece"] <= r_c["ece"] and r_r["logloss"] <= r_c["logloss"])
    print(f"\n判定: ローリング特徴は{'有用（本番統合を推奨）' if improved else '明確な改善なし（本番据え置き）'}")


if __name__ == "__main__":
    main()
