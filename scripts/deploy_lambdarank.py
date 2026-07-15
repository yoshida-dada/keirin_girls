"""本番予測モデルを LightGBM(lambdarank) に置換して保存する（A: モデル採用）。

compare_models の検証で lambdarank が PL線形より較正(ece)・logloss・三連単top10 で明確に優位
だったため、本番 pl_model.pkl を lambdarank へ置き換える。特徴は本番同様の 拡張20+rel_elo(as-of)。
GBDTModel は PLModel と同じ .strengths(X, car_numbers) 契約なので predict_race/build_predictions は
無改修で流用できる（persist.py が kind=="gbdt" を判別してロード）。

  python scripts/deploy_lambdarank.py --db data/keirin.sqlite         # 本番へ保存
  python scripts/deploy_lambdarank.py --db data/keirin.sqlite --out-dir /tmp/m   # 検証用

処理:
  1) 時系列分割で PL線形 vs lambdarank を再確認（採用の妥当性ログ）
  2) 全データで lambdarank を学習 → pl_model.pkl（kind=gbdt）
  3) elo_state.json（最終Elo）を更新（ライブ推論のrel_elo用）
API用の基本8特徴モデル(pl_model_base.pkl)はPLのまま（lightgbm非依存の軽量経路として温存）。
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
from src.model.elo import compute_pre_race_elo, final_elo_state, DEFAULT_ELO
from src.model.persist import save_model, save_elo_state, DEFAULT_MODEL_PATH, DEFAULT_ELO_STATE_PATH


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
    ap = argparse.ArgumentParser(description="本番モデルを LightGBM(lambdarank) に置換")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--out-dir", help="保存先（既定=本番 data/models）")
    args = ap.parse_args()
    model_path = (Path(args.out_dir) / "pl_model.pkl") if args.out_dir else DEFAULT_MODEL_PATH
    elo_path = (Path(args.out_dir) / "elo_state.json") if args.out_dir else DEFAULT_ELO_STATE_PATH

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    print(f"サンプル {len(base)}レース")
    pre_elo = compute_pre_race_elo(args.db)
    withelo = _augment(base, pre_elo)

    # 採用妥当性の再確認（時系列分割）
    tr, te = time_split(withelo, 0.25)
    r_pl = evaluate(train_pl(tr).strengths, te)
    r_gb = evaluate(train_gbdt(tr).strengths, te)
    print(f"\n{'指標':<10}{'PL線形':>12}{'lambdarank':>14}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{k:<10}{r_pl[k]:>12}{r_gb[k]:>14}")
    better = (r_gb["ece"] <= r_pl["ece"] and r_gb["logloss"] <= r_pl["logloss"])
    print(f"\nlambdarank が較正・logloss で優位: {better}")

    # 全データで学習して本番保存
    model = train_gbdt(withelo)
    save_model(model, model_path)
    save_elo_state(final_elo_state(args.db), elo_path)
    print(f"\n保存: {model_path.name}（LightGBM lambdarank, {len(model.feature_names)}特徴 rel_elo込み）")
    print(f"保存: {elo_path.name}（最終Elo {len(final_elo_state(args.db))}名）")

    # 保存物のロード健全性チェック（本番と同じ経路）
    from src.model.persist import load_model
    m2 = load_model(model_path)
    st = m2.strengths(withelo[-1].X, withelo[-1].car_numbers)
    print(f"ロード確認: {type(m2).__name__} / 例レースの1着確率合計={sum(st.values()):.3f}（≈1.0で健全）")


if __name__ == "__main__":
    main()
