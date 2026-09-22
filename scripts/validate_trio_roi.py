"""三連複を数点に絞った買い方の**実際の回収率**を測る（男子7車）。

`backfill_trio.py` で三連複の払戻を取得したので、推定ではなく実測できるようになった。
これまで測れたのは的中率と「三連単で等価に買った場合」だけだった。

**事前登録した採否基準（後から緩めない）**:
  主基準: レース単位ブートストラップ95%区間の下限が100%を超える点数が存在する
  副基準: その点数で5foldのうち4fold以上で単独ROI>100%
  多重比較: 点数を6通り試すので Bonferroni 補正（区間水準 1-0.05/6）を主基準に使う。
  満たさなければ三連複も打ち止め。三連単と同じ結論になる。

**比較のため三連単も同じ選び方で並べる**（三連複N点 ⇔ その6通りの三連単＝6N点）。
プールの値付けが違うかどうかがここで見える。

  PYTHONIOENCODING=utf-8 python scripts/validate_trio_roi.py
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
    trio = {r: (tuple(int(x) for x in cb.split("-")), p) for r, cb, p in
            c.execute("SELECT race_id,combo,payout FROM payouts_trio")}
    tri = {r: p for r, p in c.execute("SELECT race_id,payout FROM payouts_trifecta")}
    c.close()
    return nb, pos, sbm, trio, tri


def _lines(d):
    mem = defaultdict(list)
    for car, (li, pi) in d.items():
        mem[li].append((pi, car))
    return [[x for _, x in sorted(v)] for _, v in sorted(mem.items())]


def trio_probs(pr):
    out = defaultdict(float)
    for (a, b, c), p in pr.items():
        out[tuple(sorted((a, b, c)))] += p
    return dict(out)


def boot(rows, alpha=0.05, n_boot=3000, seed=0):
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
    return (point, vals[max(0, int(alpha / 2 * len(vals)))],
            vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))])


def main() -> None:
    ap = argparse.ArgumentParser(description="三連複の実回収率")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    nb, pos, sbm, trio, tri = _ctx(args.db)
    raw = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    smp = augment_samples(raw, args.db, men_features())

    rows = []
    for s in smp:
        d, P = nb.get(s.race_id), pos.get(s.race_id)
        if not d or not P or 1 not in P or 2 not in P or 3 not in P:
            continue
        if s.race_id not in trio:           # 三連複の払戻を持っているレースだけ
            continue
        bs = [x for x in d if sbm[s.race_id].get(x) and "B" in str(sbm[s.race_id][x])]
        if len(bs) != 1:
            continue
        rows.append((s, _lines(d), bs[0], (P[1], P[2], P[3])))
    print(f"対象 {len(rows):,}レース（三連複の払戻あり・男子7車）")
    if len(rows) < 500:
        print("※ 母数が少ない。判定は参考値")

    agg = {n: {"hit": 0, "n": 0, "rows": [], "trirows": [], "f": defaultdict(list)}
           for n in POINTS}
    for fi, (a0, b0, c0) in enumerate(fold_boundaries(len(rows), n_folds=args.folds,
                                                       warmup_frac=0.40, window="expanding")):
        tr_, te = rows[a0:b0], rows[b0:c0]
        model = train_gbdt([r[0] for r in tr_])
        btr = []
        for s, ln, bt, _o in tr_:
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
            seq = sorted(trio_probs(mix).items(), key=lambda kv: -kv[1])
            truth, yen = trio[s.race_id]
            fy = tri.get(s.race_id, 0)
            for n in POINTS:
                buy = [k for k, _ in seq[:n]]
                h = int(truth in buy)
                A = agg[n]
                A["n"] += 1; A["hit"] += h
                A["rows"].append((STAKE * n, yen if h else 0))
                A["trirows"].append((STAKE * 6 * n, fy if h else 0))
                A["f"][fi].append((STAKE * n, yen if h else 0))

    alpha = 0.05 / len(POINTS)               # Bonferroni
    print(f"走査点数 {len(POINTS)}通り → 補正後の区間水準 {(1-alpha)*100:.2f}%\n")
    print(f"{'点数':>5}{'的中率':>8}{'三連複ROI':>11}{'補正区間':>22}{'fold勝ち':>9}"
          f"{'三連単等価ROI':>15}")
    hits = []
    for n in POINTS:
        A = agg[n]
        if not A["n"]:
            continue
        hit = A["hit"] / A["n"] * 100
        p, lo, hi = boot(A["rows"], alpha)
        tp, _, _ = boot(A["trirows"], alpha)
        wins = sum(1 for f in A["f"].values()
                   if sum(x for _, x in f) > sum(a for a, _ in f))
        ok = lo is not None and lo > 1.0 and wins >= 4
        if ok:
            hits.append(n)
        print(f"{n:>5}{hit:>7.1f}%{p*100:>10.1f}%"
              f"{f'[{lo*100:.1f}–{hi*100:.1f}%]':>22}{wins:>7}/5"
              f"{tp*100:>14.1f}%" + ("  ★" if ok else ""))

    print(f"\n事前基準を満たす点数: {hits if hits else 'なし'}")
    if not hits:
        print("  → 三連複も黒字にならない。三連単と同じ結論。")
    print("※ 三連単等価ROI＝同じ三連複をその6通りの三連単で買った場合（的中条件は同一）。")
    print("  両者が近ければ賭式間で値付けが整合＝券種を変えても意味が無い、を示す。")


if __name__ == "__main__":
    main()
