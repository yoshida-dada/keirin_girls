"""統合買い目（全分岐をケアする買い目）の的中率・回収率を実測する（男子7車）。

ダッシュボードに出している `dev_branches.merged` の形をそのまま採点する。
点数を P(展開) に比例配分した複数フォーメーションで、**画面には「モデル予測 X%」と
出しているが実測値が無い**（買い目の型 ◎頭/◎2着… には walk-forward 実測を併記済み）。
ここを埋める。

**オッズ足切りは測らない。** `odds_final_trifecta` が確定オッズではなく暫定オッズと
判明しており（2026-08-17）、オッズを使う結論は保留中。ここは払戻だけで測れる
「足切り前」の買い目を対象にする。

**事前登録した採否基準（後から緩めない）**:
  主基準: モデル予測（p_model）と実測的中率の**平均絶対誤差が 3pt 以内**
          ＝画面に出している確率が信用できるか
  副基準: 予測十分位と実測率の Spearman 順位相関 >= 0.9（順序が保たれているか）
  回収率は参考値。控除率25%のため上限75%で、100%超は期待しない（黒字ゾーンは
  検証済みで存在しない）。

  PYTHONIOENCODING=utf-8 python scripts/validate_plan.py
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.model.feature_sets import men_features
from src.model.development_branches import (branch_mixture, build_plan, _combos,
                                            PLAN_POINTS)
from src.backtest.walkforward import fold_boundaries

STAKE = 100
BUDGETS = [12, 18, 24]


def _ctx(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    nb = defaultdict(dict)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi"
            " WHERE line_id IS NOT NULL"):
        nb[rid][car] = (li, pi)
    pos, sbm = defaultdict(dict), defaultdict(dict)
    for rid, p, car, s in c.execute("SELECT race_id,position,car_number,sb FROM results"):
        pos[rid][p] = car
        sbm[rid][car] = s
    pay = {r: (combo, p) for r, combo, p in
           c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    c.close()
    return nb, pos, sbm, pay


def _lines(d):
    mem = defaultdict(list)
    for car, (li, pi) in d.items():
        mem[li].append((pi, car))
    return [[x for _, x in sorted(v)] for _, v in sorted(mem.items())]


def _spearman(a, b):
    n = len(a)
    if n < 3:
        return 0.0
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    d2 = sum((ra[x] - rb[y]) ** 2 for x, y in zip(a, b))
    return 1 - 6 * d2 / (n * (n * n - 1))


def boot(rows, n_boot=3000, seed=0):
    if not rows:
        return None, None, None
    rnd = random.Random(seed)
    s = sum(a for a, _ in rows)
    point = sum(b for _, b in rows) / s if s else 0.0
    n = len(rows)
    vals = []
    for _ in range(n_boot):
        ss = rr = 0.0
        for _ in range(n):
            a, b = rows[rnd.randrange(n)]
            ss += a; rr += b
        if ss:
            vals.append(rr / ss)
    vals.sort()
    return point, vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def main() -> None:
    ap = argparse.ArgumentParser(description="統合買い目の検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    nb, pos, sbm, pay = _ctx(args.db)
    raw = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    smp = augment_samples(raw, args.db, men_features())

    rows = []
    for s in smp:
        d, P = nb.get(s.race_id), pos.get(s.race_id)
        if not d or not P or 1 not in P or 2 not in P or 3 not in P:
            continue
        if s.race_id not in pay:
            continue
        bs = [x for x in d if sbm[s.race_id].get(x) and "B" in str(sbm[s.race_id][x])]
        if len(bs) != 1:
            continue
        rows.append((s, _lines(d), bs[0], (P[1], P[2], P[3])))
    print(f"対象 {len(rows):,}レース（男子7車・B一意・3着確定・払戻あり）")
    print(f"本番の総点数は PLAN_POINTS={PLAN_POINTS}。ここでは {BUDGETS} で比較する\n")

    agg = {b: {"pred": 0.0, "hit": 0, "n": 0, "pts": 0, "rows": [],
               "cal": []} for b in BUDGETS}
    for fi, (a0, b0, c0) in enumerate(fold_boundaries(len(rows), n_folds=args.folds,
                                                       warmup_frac=0.40, window="expanding")):
        tr, te = rows[a0:b0], rows[b0:c0]
        model = train_gbdt([r[0] for r in tr])
        btr = []
        for s, ln, bt, _o in tr:
            t = type(s)(**{**s.__dict__})
            t.order = [bt] + [x for x in s.car_numbers if x != bt]
            btr.append(t)
        bmodel = train_gbdt(btr)

        for s, ln, bt, order in te:
            st = model.strengths(s.X, s.car_numbers)
            pb = bmodel.strengths(s.X, s.car_numbers)
            if not st or not pb:
                continue
            mix, dists = branch_mixture(st, ln, pb)
            if not dists:
                continue
            pm = lambda f: sum(mix.get(k, 0.0) for k in _combos(f))
            win, yen = pay[s.race_id]
            wt = tuple(int(x) for x in win.split("-"))
            for bud in BUDGETS:
                plan = build_plan(dists, mix, pm, None, total=bud)
                if not plan:
                    continue
                used = set()
                for f in plan["forms"]:
                    used |= set(_combos(f))
                if not used:
                    continue
                A = agg[bud]
                p = plan["before"]["p_model"]
                h = int(wt in used)
                A["pred"] += p
                A["hit"] += h
                A["n"] += 1
                A["pts"] += len(used)
                A["rows"].append((STAKE * len(used), yen if h else 0))
                A["cal"].append((p, h))

    print(f"{'総点数':>7}{'実点数':>8}{'R数':>8}{'モデル予測':>11}{'実測的中率':>11}"
          f"{'誤差':>8}{'回収率':>9}{'95%区間':>20}")
    maes, rhos = {}, {}
    for bud in BUDGETS:
        A = agg[bud]
        if not A["n"]:
            continue
        pred = A["pred"] / A["n"] * 100
        act = A["hit"] / A["n"] * 100
        maes[bud] = abs(pred - act)
        roi, lo, hi = boot(A["rows"])
        print(f"{bud:>7}{A['pts']/A['n']:>8.1f}{A['n']:>8,}{pred:>10.1f}%{act:>10.1f}%"
              f"{pred-act:>+8.1f}{roi*100:>8.1f}%"
              f"{f'[{lo*100:.1f}–{hi*100:.1f}%]':>20}")
        # 較正（十分位）
        cal = sorted(A["cal"])
        n = len(cal)
        px, ax = [], []
        for i in range(10):
            part = cal[i * n // 10:(i + 1) * n // 10]
            if part:
                px.append(sum(p for p, _ in part) / len(part))
                ax.append(sum(h for _, h in part) / len(part))
        rhos[bud] = _spearman(px, ax)
        print("        十分位 予測: " + " ".join(f"{x*100:.0f}" for x in px))
        print("        十分位 実測: " + " ".join(f"{x*100:.0f}" for x in ax)
              + f"   ρ={rhos[bud]:.2f}")

    mae = sum(maes.values()) / len(maes)
    rho = sum(rhos.values()) / len(rhos)
    print(f"\n平均絶対誤差 {mae:.2f}pt / 平均ρ {rho:.2f}")
    print("\n事前基準の判定:")
    print(f"  主基準（誤差3pt以内）: {'充足' if mae <= 3.0 else '不充足'}")
    print(f"  副基準（ρ>=0.9）: {'充足' if rho >= 0.9 else '不充足'}")
    print(f"\n→ {'画面の「モデル予測」は信用してよい' if mae <= 3.0 and rho >= 0.9 else '要修正'}")
    print("※ 回収率の上限は控除率25%より75%。100%超は期待しない（黒字ゾーンは存在しない）。")
    print("※ オッズ足切りは測っていない（odds_final_trifecta が暫定オッズと判明し保留中）。")


if __name__ == "__main__":
    main()
