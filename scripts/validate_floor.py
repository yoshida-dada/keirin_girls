"""展開分岐の買い目から安いオッズを切ったらどうなるか（男子7車）。

**足切りは「合成オッズを上げる」ことは定義上必ず起きる**（低い倍率の目を捨てるので
1/Σ(1/o) が上がる）。問題は**回収率が上がるか**で、そこは自明でない。
ガールズでは一度測って「合成オッズは上がるがROIは上がらない＝分散の付け替え」と
出ている（keirin-roi-strategy）。男子の展開分岐の買い目では未測定なので測る。

**事前の見立て**: 上がらない見込みが強い。安い目＝人気の目＝モデルが最も正確に
当てている帯で、そこを捨てて残すのは 4.21.2 で「予測10%に対し実測3.7%」と分かった
高配当帯だから。ただし見立てではなく数字で確かめる。

**測る形**: 分岐混合分布から本番と同じ買い目（分岐の買い目・◎頭・◎2着・◎3着・◎抜き）を
作り、確定オッズが floor 未満の目を落とす。floor は 0/5/10/20/30/50/100 倍。
落として0点になったレースはそのレースを買わない（見送り）扱いにする。

**事前登録した採否基準（後から緩めない）**:
  主基準: レース単位ブートストラップの Bonferroni補正済み区間の下限 > 100%
  副基準: 5foldのうち 4fold以上で単独ROI > 100%
  走査セル数はコード内で数えて補正に使う。片方だけなら候補としない。

  PYTHONIOENCODING=utf-8 python scripts/validate_floor.py
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
from src.model.development_branches import (branch_trifecta, formation_types,
                                            _formation, FORMATION_BUDGET, FORM_KINDS)
from src.backtest.walkforward import fold_boundaries

STAKE = 100
FLOORS = [0, 5, 10, 20, 30, 50, 100]


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
    odds = defaultdict(dict)
    for rid, combo, o in c.execute("SELECT race_id,combo,odds FROM odds_final_trifecta"):
        odds[rid][combo] = o
    pay = {r: (combo, p) for r, combo, p in
           c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    c.close()
    return nb, pos, sbm, odds, pay


def _lines(d):
    mem = defaultdict(list)
    for car, (li, pi) in d.items():
        mem[li].append((pi, car))
    return [[c for _, c in sorted(v)] for _, v in sorted(mem.items())]


def combos_of(f):
    return [f"{a}-{b}-{c}" for a in f.get("first", []) for b in f.get("second", [])
            for c in f.get("third", []) if len({a, b, c}) == 3]


def boot(rows, alpha, n_boot=3000, seed=0):
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
    ap = argparse.ArgumentParser(description="展開分岐の買い目に足切りを掛ける")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    nb, pos, sbm, odds_all, pay = _ctx(args.db)
    raw = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    samples = augment_samples(raw, args.db, men_features())

    rows = []
    for s in samples:
        d, P = nb.get(s.race_id), pos.get(s.race_id)
        if not d or not P or 1 not in P or 2 not in P or 3 not in P:
            continue
        if s.race_id not in pay or s.race_id not in odds_all:
            continue
        bs = [x for x in d if sbm[s.race_id].get(x) and "B" in str(sbm[s.race_id][x])]
        if len(bs) != 1:
            continue
        rows.append((s, _lines(d), bs[0]))
    print(f"対象 {len(rows):,}レース（男子7車・B一意・3着確定・オッズあり）")

    # acc[(買い目, floor)] = [(賭け金, 払戻)], synth の合計
    acc = defaultdict(list)
    facc = defaultdict(lambda: defaultdict(list))
    syn = defaultdict(list)
    skip = defaultdict(int)
    for fi, (a, b2, c) in enumerate(fold_boundaries(len(rows), n_folds=args.folds,
                                                     warmup_frac=0.40, window="expanding")):
        tr, te = rows[a:b2], rows[b2:c]
        model = train_gbdt([r[0] for r in tr])
        btr = []
        for s, lines, bt in tr:
            t = type(s)(**{**s.__dict__})
            t.order = [bt] + [x for x in s.car_numbers if x != bt]
            btr.append(t)
        bmodel = train_gbdt(btr)

        for s, lines, bt in te:
            st = model.strengths(s.X, s.car_numbers)
            pb = bmodel.strengths(s.X, s.car_numbers)
            if not st or not pb:
                continue
            fav = max(st, key=st.get)
            mix = defaultdict(float)
            for b, p in sorted(pb.items(), key=lambda kv: -kv[1])[:3]:
                if p < 0.05:
                    continue
                for k, v in branch_trifecta(st, b, lines).items():
                    mix[k] += p * v
            z = sum(mix.values())
            if z <= 0:
                continue
            mix = {k: v / z for k, v in mix.items()}
            od = odds_all[s.race_id]
            win, yen = pay[s.race_id]

            plans = {"分岐の買い目": _formation(mix, budget=FORMATION_BUDGET)}
            for t in formation_types(mix, fav, budget=FORMATION_BUDGET, min_prob=0.0):
                plans[t["kind"]] = t.get("formation")

            for name, f in plans.items():
                if not f:
                    continue
                base = combos_of(f)
                for fl in FLOORS:
                    keep = [k for k in base if (od.get(k) or 0) >= fl]
                    if not keep:
                        skip[(name, fl)] += 1       # 全部切れた＝見送り
                        continue
                    inv = sum(1.0 / od[k] for k in keep if od.get(k))
                    if inv > 0:
                        syn[(name, fl)].append(1.0 / inv)
                    acc[(name, fl)].append((STAKE * len(keep), yen if win in keep else 0))
                    facc[(name, fl)][fi].append((STAKE * len(keep), yen if win in keep else 0))

    cells = len(acc)
    alpha = 0.05 / cells
    print(f"走査セル数 {cells} → 補正後の区間水準 {(1-alpha)*100:.3f}%\n")
    print(f"{'買い目':>10}{'足切り':>7}{'買ったR':>8}{'見送り':>7}{'点数':>6}"
          f"{'的中率':>7}{'合成':>8}{'ROI':>8}{'補正区間':>21}{'fold勝ち':>8}")
    hits = []
    for name in ["分岐の買い目"] + list(FORM_KINDS):
        for fl in FLOORS:
            rr = acc.get((name, fl))
            if not rr:
                continue
            pts = sum(a for a, _ in rr) / len(rr) / STAKE
            hit = sum(1 for _, x in rr if x > 0) / len(rr)
            sy = sum(syn[(name, fl)]) / len(syn[(name, fl)]) if syn[(name, fl)] else 0
            p, lo, hi = boot(rr, alpha)
            wins = sum(1 for f in facc[(name, fl)].values()
                       if sum(x for _, x in f) > sum(a for a, _ in f))
            ok = lo is not None and lo > 1.0 and wins >= 4
            if ok:
                hits.append((name, fl, p, lo, hi, wins))
            print(f"{name:>10}{fl:>6}倍{len(rr):>8,}{skip.get((name,fl),0):>7,}{pts:>6.1f}"
                  f"{hit*100:>6.1f}%{sy:>7.1f}倍{p*100:>7.1f}%"
                  f"{f'[{lo*100:.1f}–{hi*100:.1f}%]':>21}{wins:>6}/5" + ("  ★" if ok else ""))
        print()

    print(f"事前基準を満たすセル: {len(hits)}")
    for h in hits:
        print(f"  ★ {h[0]} {h[1]}倍以上  ROI {h[2]*100:.1f}% "
              f"[{h[3]*100:.1f}–{h[4]*100:.1f}%] {h[5]}/5fold")
    if not hits:
        print("  → 足切りでも黒字にならない。合成オッズは上がるが回収率は上がらない。")


if __name__ == "__main__":
    main()
