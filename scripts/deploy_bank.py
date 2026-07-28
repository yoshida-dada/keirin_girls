"""バンク交互作用2列を加えた本番モデル（38特徴）を学習・デプロイする。

現行本番は36特徴（拡張20 + rel_elo + 展開10 + 並び5）。ここに
  x_esc_bank  = 逃げ残率(レース内相対) × バンクの逃げ有利度
  x_lead_bank = 主導権指数(レース内相対) × バンクの逃げ有利度
を足して38特徴にする。バンクはレース定数で順位モデルはシフト不変のため、
交互作用の形でしかバンク情報を注入できない（詳細は src/features/bank_features.py）。

walk-forward検証（scripts/validate_bank_interaction.py）で主基準を2条件とも通過:
  5fold 全体tri10 +0.92pt(4/5) / top1 +0.49pt(4/5) / 333m top1 +0.99pt(4/5)
  7fold 全体tri10 +0.66pt(5/7) / top1 +0.43pt(4/7) / 333m top1 +0.76pt(3/7)

**backstretch（展開AI）も同じ feature_names を前提に strengths_from_model を共有するため、
本スクリプトの後に必ず deploy_backstretch.py を実行すること**（さもないと推論が落ちる）。

  PYTHONIOENCODING=utf-8 python scripts/deploy_bank.py --db data/keirin.sqlite
  PYTHONIOENCODING=utf-8 python scripts/deploy_bank.py --dry-run   # 保存せず比較のみ
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate, time_split
from src.model.persist import (save_model, load_model, DEFAULT_MODEL_PATH,
                               DEFAULT_ELO_STATE_PATH, save_elo_state)
from src.model.elo import final_elo_state
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS
from src.features.bank_features import BANK_KEYS


def main() -> None:
    ap = argparse.ArgumentParser(description="バンク交互作用込み本番モデル(38特徴)のデプロイ")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--out-dir")
    ap.add_argument("--dry-run", action="store_true", help="保存せず比較だけ行う")
    args = ap.parse_args()
    model_path = (Path(args.out_dir) / "pl_model.pkl") if args.out_dir else DEFAULT_MODEL_PATH
    elo_path = (Path(args.out_dir) / "elo_state.json") if args.out_dir else DEFAULT_ELO_STATE_PATH

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    # 現行本番と同じ36特徴（t_escwin_rel は検証中の実験列なので除外する）
    feats36 = [f for f in (list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
                           + list(NARABI_KEYS)) if f != "t_escwin_rel"]
    feats38 = feats36 + list(BANK_KEYS)
    s36 = augment_samples(base, args.db, feats36)
    s38 = augment_samples(base, args.db, feats38)
    n36, n38 = len(s36[0].feature_names), len(s38[0].feature_names)
    assert n36 == len(feats36) and n38 == len(feats38), "列数が要求と不一致"
    print(f"サンプル {len(base)}レース / 現行{n36}列 / バンク込み{n38}列")

    tr36, te36 = time_split(s36, 0.25)
    tr38, te38 = time_split(s38, 0.25)
    r36 = evaluate(train_gbdt(tr36).strengths, te36)
    r38 = evaluate(train_gbdt(tr38).strengths, te38)
    print(f"\n{'指標':<10}{f'{n36}特徴':>12}{f'{n38}(+バンク)':>14}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{k:<10}{r36[k]:>12}{r38[k]:>14}")
    print("※ この単一分割は参考。採否は walk-forward（validate_bank_interaction.py）で判断済み。")

    if args.dry_run:
        print("\n--dry-run のため保存しません。")
        return

    # 旧モデルを退避（慣習: pl_model_<特徴数>feat_backup.pkl。*_backup.pkl は git 管理外）
    if model_path.exists():
        bak = model_path.with_name(f"pl_model_{n36}feat_backup.pkl")
        shutil.copy2(model_path, bak)
        print(f"\n旧モデルを退避: {bak.name}")

    model = train_gbdt(s38)
    save_model(model, model_path)
    save_elo_state(final_elo_state(args.db), elo_path)
    print(f"保存: {model_path.name}（LightGBM lambdarank, {len(model.feature_names)}特徴 バンク交互作用込み）")

    m2 = load_model(model_path)
    st = m2.strengths(s38[-1].X, s38[-1].car_numbers)
    print(f"ロード確認: {type(m2).__name__} / {len(m2.feature_names)}特徴 / 1着確率合計={sum(st.values()):.3f}")
    print("\n★ 次に必ず実行: python scripts/deploy_backstretch.py --db " + args.db)


if __name__ == "__main__":
    main()
