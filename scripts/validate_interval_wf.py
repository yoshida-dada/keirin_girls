"""出走間隔の非線形エンコードが予測に純増するかを as-of walk-forward で検証(男子)。

ベースライン=本番男子特徴(model.feature_names)。間隔=各選手の前走からの日数(全種別の
実走履歴から算出, 発走前に既知=リーク無し)。エンコード3種を各々足して比較:
  E1 線形     : min(gap,120)/30
  E2 30日超ランプ: max(0,gap-30)/30   （記述統計で45日超に残差低下＝長期ブランク狙い）
  E3 45日+フラグ : 1 if gap>=45 else 0
top1/tri10(全体) と「45日以上ブランク選手が居るレース」サブセットで純増を判定。

  PYTHONIOENCODING=utf-8 python scripts/validate_interval_wf.py --folds 5
"""
from __future__ import annotations

import argparse
import copy
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate
from src.model.plackett_luce import all_trifecta_probs
from src.backtest.walkforward import fold_boundaries

DB = str(DATA_DIR / "keirin_men.sqlite")


def _adate(rid):
    return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=int(rid[10:12]) - 1)


def _load_gap():
    """(race_id, rider_name) -> 前走からの日数（全種別の実走履歴から, as-of）。"""
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    rows = c.execute("SELECT race_id,car_number,rider_name FROM entries").fetchall()
    c.close()
    byname = defaultdict(set)
    for rid, car, nm in rows:
        byname[nm].add((_adate(rid), rid))
    gap = {}   # (rid, name) -> gap days
    for nm, s in byname.items():
        seq = sorted(s)
        prev = None
        for d, rid in seq:
            if prev is not None and (d - prev).days > 0:
                gap[(rid, nm)] = (d - prev).days
            prev = d
    return gap, {(rid, car): nm for rid, car, nm in rows}


def _tri10(model, test):
    if not test:
        return 0.0
    hit = 0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        ranked = [k for k, _ in sorted(all_trifecta_probs(st).items(), key=lambda kv: -kv[1])]
        hit += int(tuple(s.order[:3]) in ranked[:10])
    return hit / len(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    model, _, _ = load_for(False)
    base = load_samples(DB, field_size=[7, 9], features=PL_FEATURES_FULL)
    s0 = augment_samples(base, DB, model.feature_names)
    gap, name = _load_gap()

    def gap_of(s, car):
        nm = name.get((s.race_id, car))
        return gap.get((s.race_id, nm)) if nm else None

    def add(samples, fn):
        out = []
        for s in samples:
            col = np.array([[fn(gap_of(s, c))] for c in s.car_numbers], dtype=float)
            s2 = copy.copy(s)
            s2.X = np.hstack([s.X, col])
            s2.feature_names = list(s.feature_names) + ["gap_enc"]
            out.append(s2)
        return out

    encs = {
        "E1 線形min120/30": lambda g: (min(g, 120) / 30.0) if g else 0.0,
        "E2 30日超ランプ": lambda g: (max(0, g - 30) / 30.0) if g else 0.0,
        "E3 45日+フラグ": lambda g: 1.0 if (g and g >= 45) else 0.0,
    }
    variants = {nm: add(s0, fn) for nm, fn in encs.items()}
    bounds = fold_boundaries(len(s0), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"男子 walk-forward {len(bounds)}fold  ベース{len(s0)}レース  出走間隔エンコード検証\n")

    # 長期ブランク在籍レース判定
    def has_long(s):
        return any((gap_of(s, c) or 0) >= 45 for c in s.car_numbers)

    res = {nm: {"t1": [], "t10": [], "lt1": [], "lt10": []} for nm in encs}
    base_m = {"t1": [], "t10": [], "lt1": [], "lt10": []}
    for a, b, c in bounds:
        m0 = train_gbdt(s0[a:b])
        te0 = s0[b:c]
        longidx = [j for j in range(len(te0)) if has_long(te0[j])]
        c0 = evaluate(m0.strengths, te0)
        base_m["t1"].append(c0.get("top1_acc", 0)); base_m["t10"].append(_tri10(m0, te0))
        base_m["lt1"].append(evaluate(m0.strengths, [te0[j] for j in longidx]).get("top1_acc", 0) if longidx else 0)
        base_m["lt10"].append(_tri10(m0, [te0[j] for j in longidx]))
        for nm in encs:
            sv = variants[nm]
            mg = train_gbdt(sv[a:b]); teg = sv[b:c]
            cg = evaluate(mg.strengths, teg)
            res[nm]["t1"].append(cg.get("top1_acc", 0)); res[nm]["t10"].append(_tri10(mg, teg))
            res[nm]["lt1"].append(evaluate(mg.strengths, [teg[j] for j in longidx]).get("top1_acc", 0) if longidx else 0)
            res[nm]["lt10"].append(_tri10(mg, [teg[j] for j in longidx]))

    n = len(bounds)
    def mean(a): return sum(a) / len(a) if a else 0
    print(f"ベース: top1 {mean(base_m['t1'])*100:.1f}% / tri10 {mean(base_m['t10'])*100:.1f}%"
          f" / 長期在籍R top1 {mean(base_m['lt1'])*100:.1f}% tri10 {mean(base_m['lt10'])*100:.1f}%")
    print(f"\n{'エンコード':<18}{'top1Δ':>9}{'勝fold':>7}{'tri10Δ':>9}{'勝fold':>7}"
          f"{'長期top1Δ':>11}{'長期tri10Δ':>12}")
    for nm in encs:
        d1 = [res[nm]["t1"][i] - base_m["t1"][i] for i in range(n)]
        d10 = [res[nm]["t10"][i] - base_m["t10"][i] for i in range(n)]
        l1 = [res[nm]["lt1"][i] - base_m["lt1"][i] for i in range(n)]
        l10 = [res[nm]["lt10"][i] - base_m["lt10"][i] for i in range(n)]
        print(f"{nm:<18}{mean(d1)*100:>+8.2f}{sum(x>0 for x in d1):>5}/{n}"
              f"{mean(d10)*100:>+8.2f}{sum(x>0 for x in d10):>5}/{n}"
              f"{mean(l1)*100:>+10.2f}{mean(l10)*100:>+11.2f}")
    print("\n判定: 全体top1/tri10が過半fold+かつ平均Δ>0→純増(採用検討)。長期在籍Rだけ改善で全体中立なら"
          "『稀な長期ブランクの補正』として価値。全て≈0/符号ばらつき→既存特徴に吸収。")


if __name__ == "__main__":
    main()
