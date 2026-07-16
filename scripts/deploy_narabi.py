"""本番 lambdarank に並び予想3列(narabi_pos/lead/leg)を統合して再学習・保存する（34特徴）。

walk-forward検証で並び予想は「混戦の三連単top10を平均+1.7pt・全体は中立(無害)」と確認済みのため採用。
narabi は発走前確定・出走表ページから即取得できるので推論も容易（strengths_from_model が narabi_ctx で付与）。

  python scripts/deploy_narabi.py --db data/keirin.sqlite            # 本番へ保存
  python scripts/deploy_narabi.py --db data/keirin.sqlite --out-dir /tmp/m   # 検証用
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
from src.features.rider_narabi import NARABI_KEYS
from src.model.persist import save_model, save_elo_state, load_model, DEFAULT_MODEL_PATH, DEFAULT_ELO_STATE_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description="本番lambdarankに並び予想を統合(34特徴)")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    model_path = (Path(args.out_dir) / "pl_model.pkl") if args.out_dir else DEFAULT_MODEL_PATH
    elo_path = (Path(args.out_dir) / "elo_state.json") if args.out_dir else DEFAULT_ELO_STATE_PATH

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    feats34 = feats31 + list(NARABI_KEYS)
    s31 = augment_samples(base, args.db, feats31)
    s34 = augment_samples(base, args.db, feats34)
    print(f"サンプル {len(base)}レース / 31→{len(s31[0].feature_names)}列 / 34→{len(s34[0].feature_names)}列")

    tr31, te31 = time_split(s31, 0.25)
    tr34, te34 = time_split(s34, 0.25)
    r31 = evaluate(train_gbdt(tr31).strengths, te31)
    r34 = evaluate(train_gbdt(tr34).strengths, te34)
    print(f"\n{'指標':<10}{'31特徴':>12}{'34(+並び)':>12}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"{k:<10}{r31[k]:>12}{r34[k]:>12}")

    model = train_gbdt(s34)
    save_model(model, model_path)
    save_elo_state(final_elo_state(args.db), elo_path)
    print(f"\n保存: {model_path.name}（LightGBM lambdarank, {len(model.feature_names)}特徴 並び予想込み）")

    m2 = load_model(model_path)
    st = m2.strengths(s34[-1].X, s34[-1].car_numbers)
    print(f"ロード確認: {type(m2).__name__} / {len(m2.feature_names)}特徴 / 1着確率合計={sum(st.values()):.3f}")


if __name__ == "__main__":
    main()
