"""落車明け特徴が予測に純増するかを as-of walk-forward で検証(男子)。

落車固有(dnf_status='落')の明けは残差 明け1走-0.34→6走-0.15→7走で回復。これを
「落車からのレース数の減衰ランプ」で入れて top1/tri10/ECE の純増と、落車明け在籍レースの
改善を見る。エンコード: crash_recency = max(0, (R - min(rs,R)))/R （落車履歴なしは0）。

  PYTHONIOENCODING=utf-8 python scripts/validate_crash_wf.py --folds 5
"""
from __future__ import annotations

import argparse
import copy
import sqlite3
import sys
from collections import defaultdict
from datetime import date
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


def _load_since_crash():
    """(race_id, car) -> 直近の落車からのレース数（as-of）。落車履歴なしは None。"""
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    rdt = {rid: d for rid, d in c.execute("SELECT race_id,race_date FROM races")}
    rows = c.execute("SELECT race_id,car_number,rider_name FROM results"
                     " WHERE rider_name IS NOT NULL").fetchall()
    fell = set()
    for rid, nm in c.execute("SELECT race_id,rider_name FROM dnf_status WHERE status='落'"):
        fell.add((rid, nm))
    c.close()

    def pd(s):
        try:
            return date.fromisoformat(str(s))
        except (ValueError, TypeError):
            return None

    byrider = defaultdict(list)
    for rid, car, nm in rows:
        d = pd(rdt.get(rid))
        if d:
            byrider[nm].append((d, rid, car, nm))
    out = {}
    for nm, v in byrider.items():
        v.sort()
        since = None
        for d, rid, car, _ in v:
            # このレース時点(発走前)での「直近落車からのレース数」を確定してから、
            # このレースが落車なら since=0 にリセット
            out[(rid, car)] = since
            if (rid, nm) in fell:
                since = 0
            elif since is not None:
                since += 1
    return out


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
    ap.add_argument("--recover", type=int, default=7)
    args = ap.parse_args()
    R = args.recover
    model, _, _ = load_for(False)
    base = load_samples(DB, field_size=[7, 9], features=PL_FEATURES_FULL)
    s0 = augment_samples(base, DB, model.feature_names)
    since = _load_since_crash()

    def rec(rs):
        if rs is None:
            return 0.0
        return max(0.0, (R - min(rs, R)) / R)

    def has_recent(s):
        return any((since.get((s.race_id, c)) is not None and since.get((s.race_id, c)) < R)
                   for c in s.car_numbers)

    def add(samples):
        out = []
        for s in samples:
            col = np.array([[rec(since.get((s.race_id, c)))] for c in s.car_numbers])
            s2 = copy.copy(s)
            s2.X = np.hstack([s.X, col])
            s2.feature_names = list(s.feature_names) + ["crash_recency"]
            out.append(s2)
        return out

    sc = add(s0)
    bounds = fold_boundaries(len(s0), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"男子 walk-forward {len(bounds)}fold  ベース{len(s0)}R  落車明け特徴(recover={R})\n")
    print(f"{'fold':>4}{'top1 base→+落車':>18}{'tri10 base→+落車':>18}{'ece base→+落車':>18}"
          f"{'落車在籍R top1Δ':>16}")
    agg = defaultdict(list)
    for i, (a, b, c) in enumerate(bounds):
        m0, mc = train_gbdt(s0[a:b]), train_gbdt(sc[a:b])
        te0, tec = s0[b:c], sc[b:c]
        idx = [j for j in range(len(te0)) if has_recent(te0[j])]
        c0, cc = evaluate(m0.strengths, te0), evaluate(mc.strengths, tec)
        t0, tc = _tri10(m0, te0), _tri10(mc, tec)
        l0 = evaluate(m0.strengths, [te0[j] for j in idx]).get("top1_acc", 0) if idx else 0
        lc = evaluate(mc.strengths, [tec[j] for j in idx]).get("top1_acc", 0) if idx else 0
        agg["t1"].append((c0["top1_acc"], cc["top1_acc"]))
        agg["t10"].append((t0, tc)); agg["ece"].append((c0["ece"], cc["ece"]))
        agg["lt1"].append((l0, lc)); agg["nlong"].append(len(idx))
        print(f"{i:>4}{c0['top1_acc']*100:>10.1f}→{cc['top1_acc']*100:.1f}%"
              f"{t0*100:>11.1f}→{tc*100:.1f}%{c0['ece']:>11.4f}→{cc['ece']:.4f}"
              f"{(lc-l0)*100:>+14.2f}(n{len(idx)})")
    n = len(bounds)
    def md(k, low=False):
        ds = [b - a for a, b in agg[k]]
        return sum(ds) / n, sum((d < 0) if low else (d > 0) for d in ds)
    for k, lbl, low in [("t1", "top1", False), ("t10", "tri10", False),
                        ("ece", "ECE", True), ("lt1", "落車在籍R top1", False)]:
        m, w = md(k, low)
        u = "pt" if k != "ece" else ""
        v = m * 100 if k != "ece" else m
        print(f"  {lbl:<14} 平均Δ {v:+.3f}{u}  +落車優位fold {w}/{n}")
    print("\n判定: top1/tri10が過半fold+かつ平均Δ>0、ECE悪化なし → 採用。落車在籍Rで大きく改善が期待。")


if __name__ == "__main__":
    main()
