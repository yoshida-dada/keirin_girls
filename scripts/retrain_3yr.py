"""3年データで本番モデルを再学習し、Eloが有用なら特徴量に加えて保存する（②の仕上げ）。

  python scripts/retrain_3yr.py --db data/keirin.sqlite            # 本番モデルを更新
  python scripts/retrain_3yr.py --db data/keirin_v3.sqlite --out-dir /tmp/m   # 検証用

処理:
  1) 拡張20特徴 vs 拡張+Elo(21) を時系列検証で比較
  2) Eloが改善(ECE/logloss↓)なら rel_elo を採用
  3) 採用構成で全データ学習 → pl_model.pkl（Elo採用時は elo_state.json も）
  4) API用の基本8特徴モデル pl_model_base.pkl も更新
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES, PL_FEATURES_FULL
from src.model.train_pl import train_pl
from src.model.evaluate import evaluate, time_split
from src.model.elo import compute_pre_race_elo, final_elo_state, DEFAULT_ELO
from src.model.persist import (
    save_model, save_elo_state, DEFAULT_MODEL_PATH, DEFAULT_ELO_STATE_PATH,
)


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
    ap = argparse.ArgumentParser(description="3年データで本番モデル再学習（Elo込み）")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--out-dir", help="モデル保存先（既定=本番 data/models）")
    args = ap.parse_args()
    model_path = (Path(args.out_dir) / "pl_model.pkl") if args.out_dir else DEFAULT_MODEL_PATH
    base_path = model_path.parent / "pl_model_base.pkl"
    elo_path = (Path(args.out_dir) / "elo_state.json") if args.out_dir else DEFAULT_ELO_STATE_PATH

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    print(f"サンプル {len(base)}レース（3年想定）")
    pre_elo = compute_pre_race_elo(args.db)
    withelo = _augment(base, pre_elo)

    tr_b, te_b = time_split(base, 0.25)
    tr_e, te_e = time_split(withelo, 0.25)
    r_b = evaluate(train_pl(tr_b).strengths, te_b)
    r_e = evaluate(train_pl(tr_e).strengths, te_e)
    print(f"\n{'指標':<10}{'拡張のみ':>12}{'拡張+Elo':>12}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{k:<10}{r_b[k]:>12}{r_e[k]:>12}")

    adopt_elo = (r_e["ece"] <= r_b["ece"] and r_e["logloss"] <= r_b["logloss"])
    print(f"\nElo採用: {adopt_elo}")

    # 採用構成で全データ学習して保存
    final_samples = withelo if adopt_elo else base
    model = train_pl(final_samples)
    save_model(model, model_path)
    if adopt_elo:
        save_elo_state(final_elo_state(args.db), elo_path)
        print(f"保存: {model_path.name}（{len(model.feature_names)}特徴, rel_elo込み）＋ {elo_path.name}")
    else:
        # Elo不採用時は古い elo_state を残さない（feature_namesにrel_eloが無ければ未使用）
        print(f"保存: {model_path.name}（{len(model.feature_names)}特徴, Eloなし）")

    # API用の基本モデル（生入力で駆動可・Eloなし）
    base_model = train_pl(load_samples(args.db, features=PL_FEATURES))
    save_model(base_model, base_path)
    print(f"保存: {base_path.name}（基本{len(base_model.feature_names)}特徴, API用）")


if __name__ == "__main__":
    main()
