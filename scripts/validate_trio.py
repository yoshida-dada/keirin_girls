"""三連複を数点に絞った場合の的中率と、黒字化に必要な配当（男子7車）。

**測れること / 測れないこと**:
  測れる … 的中率（三連複の的中条件は三連単分布から一意に決まる。6通りの順列を足すだけ）
          三連単で等価に買った場合の回収率（三連複1点＝その6通りの三連単。的中条件が同一）
          黒字化に必要な三連複の平均配当
  測れない … **三連複そのものの回収率**。DBに payouts_trifecta しか無く、
            三連複の払戻を持っていない（S1で追加収集が要る）

**理論上の制約（先に押さえる）**: 市場が賭式間で整合していれば、三連複の配当は
6通りをドッチングした合成オッズ 1/Σ(1/o) と一致し、回収率は三連単と同じになる。
つまり三連複に妙味があるのは**プールの値付けが三連単と違う場合だけ**。
ここで出す「必要配当」は、その差がどれだけ必要かを数字にするためのもの。

**事前登録した判定（後から緩めない）**:
  「三連複に可能性がある」と言えるのは、**黒字化に必要な平均配当が、三連複の
  実勢配当として現実的な水準**である場合のみ。必要配当が実勢の何倍にもなるなら、
  プールの値付け差では埋まらないので追加収集をしても無駄になる。

  PYTHONIOENCODING=utf-8 python scripts/validate_trio.py
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.model.feature_sets import men_features
from src.model.development_branches import branch_mixture
from src.backtest.walkforward import fold_boundaries

STAKE = 100
POINTS = [1, 2, 3, 5, 8, 12]


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


def trio_probs(pr: dict) -> dict:
    """三連単分布 → 三連複（順不同）の確率。6通りの順列を足すだけ。"""
    out: dict = defaultdict(float)
    for (a, b, c), p in pr.items():
        out[tuple(sorted((a, b, c)))] += p
    return dict(out)


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
    ap = argparse.ArgumentParser(description="三連複を絞った場合の的中率と必要配当")
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
    print(f"対象 {len(rows):,}レース（男子7車）  三連複は7車で35通り\n")

    agg = {n: {"hit": 0, "n": 0, "pred": 0.0, "tri_rows": [], "pay": []}
           for n in POINTS}
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
            mix, _ = branch_mixture(st, ln, pb)
            if not mix:
                continue
            tp = trio_probs(mix)
            seq = sorted(tp.items(), key=lambda kv: -kv[1])
            truth = tuple(sorted(order))
            _win, yen = pay[s.race_id]
            for n in POINTS:
                buy = [k for k, _ in seq[:n]]
                h = int(truth in buy)
                A = agg[n]
                A["n"] += 1
                A["hit"] += h
                A["pred"] += sum(p for _, p in seq[:n])
                # 三連単で等価に買う: 1三連複あたり6通り＝6点
                A["tri_rows"].append((STAKE * 6 * n, yen if h else 0))
                if h:
                    A["pay"].append(yen)

    print(f"{'点数':>5}{'モデル予測':>11}{'実測的中率':>11}{'誤差':>8}"
          f"{'三連単等価の回収率':>20}{'必要な三連複配当':>18}{'的中時の三連単配当(中央)':>24}")
    for n in POINTS:
        A = agg[n]
        if not A["n"]:
            continue
        pred = A["pred"] / A["n"] * 100
        hit = A["hit"] / A["n"] * 100
        roi, lo, hi = boot(A["tri_rows"])
        # 黒字化に必要な三連複の平均配当: 点数×100 / 的中率
        need = (n * STAKE) / (A["hit"] / A["n"]) if A["hit"] else float("inf")
        med = sorted(A["pay"])[len(A["pay"]) // 2] if A["pay"] else 0
        print(f"{n:>5}{pred:>10.1f}%{hit:>10.1f}%{pred-hit:>+8.1f}"
              f"{roi*100:>13.1f}% [{lo*100:.0f}–{hi*100:.0f}%]"
              f"{need:>17,.0f}円{med:>21,}円")

    print("\n※ 三連単等価の回収率＝三連複1点をその6通りの三連単で買った場合（的中条件は同一）。")
    print("※ 必要な三連複配当＝黒字化に必要な平均配当（点数×100円 ÷ 的中率）。")
    print("※ 市場が賭式間で整合していれば三連複の配当は6通りの合成オッズと一致し、")
    print("   回収率は三連単と同じになる。妙味があるのはプールの値付けが違う場合だけ。")
    print("※ **三連複そのものの回収率は測れない**（払戻データが無い。S1で追加収集が要る）。")


if __name__ == "__main__":
    main()
