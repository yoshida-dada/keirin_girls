"""LEG_AGGR の修正が悪化を招かないかを walk-forward で検証（ガールズ本番の不具合修正）。

旧版の LEG_AGGR は実データの語彙を網羅しておらず、**ガールズは33.9%の値が既定1.0
（自在扱い）にフォールバック**していた（「追上」28.1%が主因。実測B取得率は1.6%なので
本来0が妥当）。`narabi_leg` / `narabi_midleg` は本番38特徴モデルに入っているため実害がある。

ただし**書き換えるだけでは train/inference skew になる**（学習時と推論時で値が変わる）ので、
再学習とセットでしか直せない。本スクリプトは「直した特徴で学習し直すと悪化しないか」を見る。

**事前登録した採否基準（後から緩めない）**:
  これは新機能ではなく**不具合修正**なので「良くなること」ではなく「悪化しないこと」を条件にする。
  採用: top1 と tri10 の 5fold平均がどちらも悪化しない（>= 0）
  悪化した場合は現状維持し、なぜ誤った符号化の方が良かったのかを調べる（採用しない）。

  PYTHONIOENCODING=utf-8 python scripts/validate_leg_aggr_fix.py
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate
from src.model.feature_augment import augment_samples
from src.model.plackett_luce import all_trifecta_probs
from src.model.feature_sets import girls_features
from src.features.rider_narabi import compute_narabi_features, LEG_AGGR, NARABI_KEYS
from src.backtest.walkforward import fold_boundaries

# 修正前のマッピング（比較用に固定して持つ。実データ語彙の33.9%が未定義だった）
LEG_AGGR_OLD = {"先行": 2, "押え先": 2, "捲り": 2, "自在": 1,
                "追込": 0, "差し": 0, "マーク": 0, "追": 0}


def _tri10(model, test) -> float:
    if not test:
        return 0.0
    hit = 0
    for s in test:
        r = sorted(all_trifecta_probs(model.strengths(s.X, s.car_numbers)).items(),
                   key=lambda kv: -kv[1])[:10]
        hit += int(tuple(s.order[:3]) in [k for k, _ in r])
    return hit / len(test)


def _with_mapping(samples, db: str, mapping: dict):
    """narabi_leg / narabi_midleg を指定マッピングで作り直した samples を返す。

    他の列は触らない。差分がこの2列だけになるようにして、修正の効果を切り出す。
    """
    import sqlite3
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    raw = {(rid, car): (pos, leg) for rid, car, pos, leg
           in c.execute("SELECT race_id,car_number,position,leg FROM narabi")}
    c.close()

    out = []
    for s in samples:
        names = list(s.feature_names)
        i_leg = names.index("narabi_leg")
        i_mid = names.index("narabi_midleg")
        X = s.X.copy()
        for r, car in enumerate(s.car_numbers):
            pos_leg = raw.get((s.race_id, car))
            if pos_leg is None:
                continue
            pos, leg = pos_leg
            a = float(mapping.get(leg, 1))
            mid = 1.0 if 2 <= pos <= 4 else 0.0
            X[r, i_leg] = a
            X[r, i_mid] = mid * a
        s2 = copy.copy(s)
        s2.X = X
        out.append(s2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="LEG_AGGR 修正の検証（ガールズ）")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    feats = girls_features()
    base = load_samples(args.db, features=PL_FEATURES_FULL)
    s_new = augment_samples(base, args.db, feats)      # 現行コード＝修正後マッピング
    s_old = _with_mapping(s_new, args.db, LEG_AGGR_OLD)
    print(f"ガールズ {len(s_new):,}R / {len(feats)}特徴")

    # 差分が narabi_leg / narabi_midleg の2列だけであることを確認
    names = list(s_new[0].feature_names)
    diff_cols = {names[j] for s1, s2 in zip(s_new[:200], s_old[:200])
                 for j in range(len(names)) if not np.allclose(s1.X[:, j], s2.X[:, j])}
    print(f"  新旧で差が出る列: {sorted(diff_cols)}")
    assert diff_cols <= {"narabi_leg", "narabi_midleg"}, "想定外の列が変化している"
    nz = sum(1 for s1, s2 in zip(s_new, s_old) if not np.allclose(s1.X, s2.X))
    print(f"  値が変わったレース: {nz:,}/{len(s_new):,}")
    if nz == 0:
        raise SystemExit("★ 新旧で値が変わっていない。マッピングの適用に失敗している。")

    bounds = fold_boundaries(len(base), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"\nwalk-forward {len(bounds)}fold\n")
    print(f"{'fold':>4}{'testR':>7}{'  top1 旧→新':>20}{'  tri10 旧→新':>21}{'   ece 旧→新':>20}")
    agg = []
    for i, (a, b, c) in enumerate(bounds):
        mo, mn = train_gbdt(s_old[a:b]), train_gbdt(s_new[a:b])
        to, tn = s_old[b:c], s_new[b:c]
        eo, en = evaluate(mo.strengths, to), evaluate(mn.strengths, tn)
        ro, rn = _tri10(mo, to), _tri10(mn, tn)
        agg.append({"top1": en["top1_acc"] - eo["top1_acc"], "tri10": rn - ro,
                    "ece": en["ece"] - eo["ece"], "ll": en["logloss"] - eo["logloss"]})
        print(f"{i:>4}{len(to):>7}{eo['top1_acc']*100:>12.1f}→{en['top1_acc']*100:.1f}%"
              f"{ro*100:>13.1f}→{rn*100:.1f}%{eo['ece']:>12.4f}→{en['ece']:.4f}")

    n = len(agg)
    av = lambda k: sum(r[k] for r in agg) / n
    wins = lambda k, low=False: sum(1 for r in agg if (r[k] < 0 if low else r[k] > 0))
    print(f"\n【修正の効果（新−旧）】{n}fold平均・+勝ちfold数")
    print(f"  top1    {av('top1')*100:+.2f}pt  勝ち {wins('top1')}/{n}")
    print(f"  tri10   {av('tri10')*100:+.2f}pt  勝ち {wins('tri10')}/{n}")
    print(f"  logloss {av('ll'):+.4f}   改善 {wins('ll', True)}/{n}")
    print(f"  ece     {av('ece'):+.5f}  改善 {wins('ece', True)}/{n}")
    ok = av("top1") >= 0 and av("tri10") >= 0
    print(f"\n  事前基準（top1・tri10とも悪化しない）: {'充足＝修正を採用し再学習' if ok else '未充足＝現状維持'}")


if __name__ == "__main__":
    main()
