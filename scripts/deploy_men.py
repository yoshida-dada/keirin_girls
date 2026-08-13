"""【Phase 3】男子競輪モデル（39特徴）の学習・評価・デプロイ。

特徴セットは src/model/feature_sets.men_features()（拡張20 + rel_elo + ライン8 + 展開10）。
決定の根拠は men_keirin_plan.md 4.9節（ライン8列が tri10 +4.61pt・5/5fold）。

ガールズモデル（data/models/pl_model.pkl）とは**別ファイル**に保存する:
  data/models/pl_model_men.pkl / elo_state_men.json
男女は同一レースを走らず Elo も選手プールも交わらないため、混ぜる意味がない。

較正も同時に評価する。検証で ece 0.00573→0.00810 と悪化しており（ガールズ本番≒0.004の2倍）、
本番投入の可否は較正を見てから判断する必要があるため、信頼度カーブを必ず出す。

  PYTHONIOENCODING=utf-8 python scripts/deploy_men.py --dry-run   # 保存せず評価のみ
  PYTHONIOENCODING=utf-8 python scripts/deploy_men.py             # 学習して保存
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate, time_split
from src.model.persist import save_model, load_model, save_elo_state
from src.model.elo import final_elo_state
from src.model.feature_augment import augment_samples
from src.model.plackett_luce import all_trifecta_probs
from src.model.feature_sets import men_features, MEN_MODEL_NAME, MEN_ELO_NAME
from src.features.line_features import LINE_KEYS, line_columns, class_level
from src.features.tactics_features import TACTIC_NAMES

MODELS = ROOT / "data" / "models"
DEFAULT_DB = ROOT / "data" / "keirin_men.sqlite"



def build_samples(db: str, field_size):
    """男子39特徴のサンプルを作る（学習・評価で共通。列順は men_features() と一致）。

    ライン列の付与も列順の整合も `augment_samples` が持つ。以前はここで列を挟み直して
    いたが、同じ処理を二重に持った結果 analyze_dev_patterns が男子モデルを使うと
    31列 vs 39列で落ちた。付与は必ず共通関数を通す（推論との skew 防止も同じ理由）。
    """
    base = load_samples(db, field_size=field_size, features=PL_FEATURES_FULL)
    return augment_samples(base, db, men_features())


def _tri10(model, test) -> float:
    if not test:
        return 0.0
    hit = 0
    for s in test:
        r = sorted(all_trifecta_probs(model.strengths(s.X, s.car_numbers)).items(),
                   key=lambda kv: -kv[1])[:10]
        hit += int(tuple(s.order[:3]) in [k for k, _ in r])
    return hit / len(test)


def calibration_report(model, test) -> None:
    """信頼度カーブ。予測1着確率の帯ごとに実際の1着率を出す（較正段の要否判断）。"""
    bins = defaultdict(lambda: [0, 0])
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        if not st:
            continue
        win = s.order[0]
        for car, p in st.items():
            b = min(9, int(p * 10))
            bins[b][0] += 1
            bins[b][1] += int(car == win)
    print(f"\n【信頼度カーブ】予測1着確率帯ごとの実際の1着率")
    print(f"  {'予測':>10}{'n':>8}{'実測':>9}{'ズレ':>8}")
    for b in sorted(bins):
        n, w = bins[b]
        if n < 50:
            continue
        mid = b * 10 + 5
        act = w / n * 100
        print(f"  {b*10:>3}-{b*10+10:>3}%{n:>8,}{act:>8.1f}%{act-mid:>+7.1f}")
    print("  （ズレが小さいほど較正が良い。系統的に+なら過小、−なら過信）")


def main() -> None:
    ap = argparse.ArgumentParser(description="男子モデル(39特徴)の学習・デプロイ")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--field", default="7", help="車立て: 7 / 7,9 / all")
    ap.add_argument("--dry-run", action="store_true", help="保存せず評価だけ")
    ap.add_argument("--out-dir", default=str(MODELS))
    args = ap.parse_args()

    fs = None if args.field == "all" else [int(x) for x in args.field.split(",")]
    feats = men_features()
    samples = build_samples(args.db, fs)
    assert list(samples[0].feature_names) == feats, \
        f"列順が men_features() と不一致:\n  実際 {samples[0].feature_names}\n  期待 {feats}"
    print(f"男子 {len(samples):,}R（車立て {args.field}） / {len(feats)}特徴")

    tr, te = time_split(samples, 0.25)
    m = train_gbdt(tr)
    r = evaluate(m.strengths, te)
    print(f"\n【hold-out評価】学習{len(tr):,}R → 検証{len(te):,}R")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"  {k:<10}{r[k]}")
    print(f"  tri10     {_tri10(m, te):.4f}")
    calibration_report(m, te)

    if args.dry_run:
        print("\n--dry-run のため保存しません。")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mp = out / MEN_MODEL_NAME
    if mp.exists():
        bak = mp.with_name(f"pl_model_men_{len(feats)}feat_backup.pkl")
        shutil.copy2(mp, bak)
        print(f"\n旧モデルを退避: {bak.name}")

    full = train_gbdt(samples)
    save_model(full, mp)
    save_elo_state(final_elo_state(args.db), out / MEN_ELO_NAME)
    print(f"保存: {mp.name}（{len(full.feature_names)}特徴）/ {MEN_ELO_NAME}")

    m2 = load_model(mp)
    st = m2.strengths(samples[-1].X, samples[-1].car_numbers)
    print(f"ロード確認: {type(m2).__name__} / {len(m2.feature_names)}特徴 / "
          f"1着確率合計={sum(st.values()):.3f}")
    print("\n★ ガールズモデル(pl_model.pkl)には触れていません。")


if __name__ == "__main__":
    main()
