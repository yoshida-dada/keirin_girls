"""本番 lambdarank に展開特徴(A+B 10列)を統合して再学習・保存する。

compare_tactics の検証で LGB +A+B が全体で logloss/ece/三連単top10 いずれも改善したため、本番を
21特徴(拡張20+rel_elo) → 31特徴(+展開10)へ更新する。特徴付与は `feature_augment.augment_samples`
（推論の `strengths_from_model` と同一の `tactic_columns` を経由）で行い train/inference skew を防ぐ。

  python scripts/deploy_tactics.py --db data/keirin.sqlite            # 本番へ保存
  python scripts/deploy_tactics.py --db data/keirin.sqlite --out-dir /tmp/m   # 検証用

処理: 1) 時系列分割で 21 vs 31 を再確認  2) 全データで31特徴lambdarankを学習→pl_model.pkl
      3) elo_state.json 更新（展開のhistory系は推論時 current_tactics で毎回算出＝state不要）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate, time_split
from src.model.elo import final_elo_state
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.model.persist import save_model, save_elo_state, load_model, DEFAULT_MODEL_PATH, DEFAULT_ELO_STATE_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description="本番lambdarankに展開特徴を統合")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--out-dir", help="保存先（既定=本番 data/models）")
    args = ap.parse_args()
    model_path = (Path(args.out_dir) / "pl_model.pkl") if args.out_dir else DEFAULT_MODEL_PATH
    elo_path = (Path(args.out_dir) / "elo_state.json") if args.out_dir else DEFAULT_ELO_STATE_PATH

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    print(f"サンプル {len(base)}レース")
    feats21 = list(PL_FEATURES_FULL) + ["rel_elo"]
    feats31 = feats21 + list(TACTIC_NAMES)
    s21 = augment_samples(base, args.db, feats21)
    s31 = augment_samples(base, args.db, feats31)
    print(f"特徴: 21={len(s21[0].feature_names)}列 / 31={len(s31[0].feature_names)}列  展開10列={TACTIC_NAMES}")

    # 採用妥当性の再確認（時系列分割）
    tr21, te21 = time_split(s21, 0.25)
    tr31, te31 = time_split(s31, 0.25)
    r21 = evaluate(train_gbdt(tr21).strengths, te21)
    r31 = evaluate(train_gbdt(tr31).strengths, te31)
    print(f"\n{'指標':<10}{'現行21':>12}{'展開込31':>12}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{k:<10}{r21[k]:>12}{r31[k]:>12}")
    better = (r31["ece"] <= r21["ece"] + 1e-9 and r31["logloss"] < r21["logloss"])
    print(f"\n展開込みが較正非悪化かつlogloss改善: {better}")

    # 全データで学習して本番保存
    model = train_gbdt(s31)
    save_model(model, model_path)
    save_elo_state(final_elo_state(args.db), elo_path)
    print(f"\n保存: {model_path.name}（LightGBM lambdarank, {len(model.feature_names)}特徴 展開込み）")
    print(f"保存: {elo_path.name}")

    # ロード健全性（本番と同じ経路）
    m2 = load_model(model_path)
    st = m2.strengths(s31[-1].X, s31[-1].car_numbers)
    print(f"ロード確認: {type(m2).__name__} / {len(m2.feature_names)}特徴 / 例レース1着確率合計={sum(st.values()):.3f}")


if __name__ == "__main__":
    main()
