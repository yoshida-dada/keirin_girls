"""展開AI（Stage1=最終バック先頭B予測）を全データで学習し保存する。

着順モデルと同じ36特徴で lambdarank を学習（order=[B取得車]）。推論は着順モデルと同一の
strengths_from_model で呼べる（softmax=P(B)）。着順モデルには影響しない（別ファイル保存）。

  python scripts/deploy_backstretch.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS
from src.model.persist import save_model, load_model
from src.model.backstretch import b_taker, as_border, BACKSTRETCH_PATH


def main():
    ap = argparse.ArgumentParser(description="展開AI(最終バック先頭B)を学習・保存")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES) + list(NARABI_KEYS)
    samples = augment_samples(base, args.db, feats)
    btk = b_taker(args.db)
    bsamples = as_border(samples, btk)
    print(f"サンプル {len(samples)}レース / B一意 {len(bsamples)}レース / 特徴 {len(bsamples[0].feature_names)}列")

    model = train_gbdt(bsamples)
    save_model(model, BACKSTRETCH_PATH)
    print(f"保存: {BACKSTRETCH_PATH.name}（展開AI, {len(model.feature_names)}特徴）")

    m2 = load_model(BACKSTRETCH_PATH)
    st = m2.strengths(bsamples[-1].X, bsamples[-1].car_numbers)
    print(f"ロード確認: {type(m2).__name__} / P(B)合計={sum(st.values()):.3f} / "
          f"推定主導権={max(st, key=st.get)}番")


if __name__ == "__main__":
    main()
