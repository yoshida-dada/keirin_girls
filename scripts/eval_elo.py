"""Elo特徴量の有用性を評価する（②）。拡張特徴量に rel_elo を足して改善するか比較する。

  python scripts/eval_elo.py --db data/keirin.sqlite

rel_elo = 発走前Elo − レース内平均Elo（レース内変動＝PLに効く形）。
改善（logloss/brier/ece↓, top1↑）すれば特徴量に採用する。
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


def _augment_with_elo(samples, pre_elo):
    out = []
    for s in samples:
        s2 = copy.copy(s)
        elos = np.array([pre_elo.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
        rel = (elos - elos.mean()).reshape(-1, 1)
        s2.X = np.hstack([s.X, rel])
        s2.feature_names = list(s.feature_names) + ["rel_elo"]
        out.append(s2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Elo特徴量の評価")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--k", type=float, default=24.0)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    print(f"サンプル {len(base)}（特徴 {len(PL_FEATURES_FULL)} / +Eloで {len(PL_FEATURES_FULL)+1}）")
    pre_elo = compute_pre_race_elo(args.db, k=args.k)
    withelo = _augment_with_elo(base, pre_elo)

    tr_b, te_b = time_split(base, 0.25)
    tr_e, te_e = time_split(withelo, 0.25)
    m_b = train_pl(tr_b)
    m_e = train_pl(tr_e)
    r_b = evaluate(m_b.strengths, te_b)
    r_e = evaluate(m_e.strengths, te_e)

    print(f"\n{'指標':<10}{'拡張のみ':>12}{'拡張+Elo':>12}")
    for key in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{key:<10}{r_b[key]:>12}{r_e[key]:>12}")
    # rel_elo の学習重み
    w = dict(zip(m_e.feature_names, m_e.weights))
    print(f"\nrel_elo の重み: {w.get('rel_elo'):+.4f}（racing_score {w.get('racing_score'):+.3f} 参考）")
    better = (r_e["ece"] <= r_b["ece"] and r_e["logloss"] <= r_b["logloss"])
    print(f"\n判定: Eloは{'有用（採用推奨）' if better else '明確な改善なし（保留）'}")


if __name__ == "__main__":
    main()
