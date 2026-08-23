"""ギア×並び位置の交互作用が予測に純増するかを as-of walk-forward で検証(男子)。

ベースライン=本番男子特徴(生gear_ratio込み, model.feature_names)。これに「位置別の相対ギア」
4列 g_lead/g_mate/g_third/g_solo（relgear=gear−レース平均gear を各位置でゲート）を足し、
top1精度・三連単top10 が純増するか、先頭サブセットで効くかをfold横断で見る。
記述統計A2は先頭×低ギアが+3.6pt(1着)だったが交絡の可能性→純増検証で採否判断。

  PYTHONIOENCODING=utf-8 python scripts/validate_gear_position_wf.py --folds 5
"""
from __future__ import annotations

import argparse
import copy
import sqlite3
import sys
from collections import defaultdict
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
GEAR_KEYS = ["g_lead", "g_mate", "g_third", "g_solo"]


def _load_pos_gear():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    gear = {(rid, car): g for rid, car, g in
            c.execute("SELECT race_id,car_number,gear_ratio FROM entries") if g is not None}
    linesz = defaultdict(lambda: defaultdict(int))
    pos = {}
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        pos[(rid, car)] = (li, pi)
        linesz[rid][li] += 1
    c.close()
    return gear, pos, linesz


def _posclass(rid, car, pos, linesz):
    pc = pos.get((rid, car))
    if pc is None:
        return None
    li, pi = pc
    if linesz[rid].get(li, 1) == 1:
        return "単騎"
    return "先頭" if pi == 0 else ("番手" if pi == 1 else "三番手")


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
    s0 = augment_samples(base, DB, model.feature_names)     # ベースライン(生gear込み)
    gear, pos, linesz = _load_pos_gear()

    def add_gear(samples):
        out = []
        idxmap = {"先頭": 0, "番手": 1, "三番手": 2, "単騎": 3}
        for s in samples:
            gs = [gear.get((s.race_id, c)) for c in s.car_numbers]
            present = [g for g in gs if g is not None]
            mean = sum(present) / len(present) if present else 0.0
            cols = np.zeros((len(s.car_numbers), 4))
            for r, c in enumerate(s.car_numbers):
                g = gear.get((s.race_id, c))
                pcl = _posclass(s.race_id, c, pos, linesz)
                if g is not None and pcl in idxmap:
                    cols[r, idxmap[pcl]] = g - mean            # 位置別 相対ギア
            s2 = copy.copy(s)
            s2.X = np.hstack([s.X, cols])
            s2.feature_names = list(s.feature_names) + GEAR_KEYS
            out.append(s2)
        return out

    sg = add_gear(s0)
    bounds = fold_boundaries(len(s0), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"男子 walk-forward {len(bounds)}fold(expanding, warmup40%)  ベース{len(s0)}レース\n")
    print(f"{'fold':>4}{'検証期間':>22}{'全体top1 base→+gear':>22}{'全体tri10 base→+gear':>22}"
          f"{'先頭top1 base→+gear':>22}")

    agg = []
    for i, (a, b, c) in enumerate(bounds):
        m0, mg = train_gbdt(s0[a:b]), train_gbdt(sg[a:b])
        te0, teg = s0[b:c], sg[b:c]
        c0, cg = evaluate(m0.strengths, te0), evaluate(mg.strengths, teg)
        t0, tg = _tri10(m0, te0), _tri10(mg, teg)
        # 先頭(逃)サブセット: 1着車が先頭のレースで top1 を見る…ではなく、各レースの先頭選手を
        # 当てられるかは別問題。ここは「先頭選手が1着のレース」に絞ってtop1精度を見る。
        lead1 = [j for j in range(len(te0)) if _posclass(te0[j].race_id, te0[j].order[0], pos, linesz) == "先頭"]
        l0 = evaluate(m0.strengths, [te0[j] for j in lead1]) if lead1 else {}
        lg = evaluate(mg.strengths, [teg[j] for j in lead1]) if lead1 else {}
        agg.append((c0.get("top1_acc", 0), cg.get("top1_acc", 0), t0, tg,
                    l0.get("top1_acc", 0), lg.get("top1_acc", 0), len(lead1)))
        d0, d1 = te0[0].date, te0[-1].date
        print(f"{i:>4}{d0+'〜'+d1:>22}"
              f"{c0.get('top1_acc',0)*100:>11.1f}→{cg.get('top1_acc',0)*100:.1f}%"
              f"{t0*100:>12.1f}→{tg*100:.1f}%"
              f"{l0.get('top1_acc',0)*100:>12.1f}→{lg.get('top1_acc',0)*100:.1f}%")

    n = len(agg)
    def mean(idx0, idx1): return sum(r[idx1] - r[idx0] for r in agg) / n
    def wins(idx0, idx1): return sum(1 for r in agg if r[idx1] > r[idx0])
    print(f"\n全体 top1 : +gearが勝ったfold {wins(0,1)}/{n} / 平均Δ {mean(0,1)*100:+.2f}pt")
    print(f"全体 tri10: +gearが勝ったfold {wins(2,3)}/{n} / 平均Δ {mean(2,3)*100:+.2f}pt")
    print(f"先頭1着R top1: +gearが勝ったfold {wins(4,5)}/{n} / 平均Δ {mean(4,5)*100:+.2f}pt")
    print("\n判定: 大半foldで + かつ 平均Δ>0 → 純増あり(採用検討)。符号ばらつき/≈0 → 交絡で生ギアに吸収済み。")


if __name__ == "__main__":
    main()
