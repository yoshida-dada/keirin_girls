"""モデルの精度改善は回収率に伝わるか（男子7車）。

**なぜ測るか**: 分岐混合で予測精度は大きく上がった（tri10 +5.85pt、ライン決着の
ズレ −20.3pt→+1.2pt）が、回収率が動いた形跡がない。買い目側は既に打ち止め
（どう切っても71%前後で平坦）なので、**モデル改善に投資する価値があるか**を
決着させる必要がある。

**同じ選び方**で分布だけ差し替えて比べる（ここを揃えないと比較にならない）:
  pl    素のPL（＝Harville）
  himo  紐補正（配線前の本表示）
  mix   分岐混合（現在の本表示）
選び方は確率降順に累積が目標に届くまで買う（10/20/30%）。払戻で回収率を出す。
オッズは選定に使わない（odds_final_trifecta が暫定オッズと判明し保留中のため）。

**事前登録した判定（後から緩めない）**:
  「モデル改善はROIに伝わる」と言えるのは、**mix の回収率が pl より
  レース単位ブートストラップ95%区間で有意に高い**場合のみ
  （区間が重なるなら「伝わらない」と読む）。
  精度が大きく上がった変更でROIが動かないなら、モデル改善の投資対効果は無い。

  PYTHONIOENCODING=utf-8 python scripts/validate_model_to_roi.py
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
from src.model.plackett_luce import all_trifecta_probs
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.development_branches import branch_mixture
from src.backtest.walkforward import fold_boundaries

STAKE = 100
TARGETS = [0.10, 0.20, 0.30]


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


def pick(pr, target):
    """確率降順に累積が target に届くまで買う。"""
    out, cum = [], 0.0
    for k, p in sorted(pr.items(), key=lambda kv: -kv[1]):
        out.append(k)
        cum += p
        if cum >= target:
            break
    return out


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
    ap = argparse.ArgumentParser(description="モデル改善はROIに伝わるか")
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
        rows.append((s, d, _lines(d), bs[0]))
    print(f"対象 {len(rows):,}レース（男子7車）")

    names = ["pl", "himo", "mix"]
    acc = {(k, t): [] for k in names for t in TARGETS}
    hits = {(k, t): [0, 0] for k in names for t in TARGETS}
    tri10 = {k: [0, 0] for k in names}
    for fi, (a0, b0, c0) in enumerate(fold_boundaries(len(rows), n_folds=args.folds,
                                                       warmup_frac=0.40, window="expanding")):
        tr, te = rows[a0:b0], rows[b0:c0]
        model = train_gbdt([r[0] for r in tr])
        btr = []
        for s, d, ln, bt in tr:
            t = type(s)(**{**s.__dict__})
            t.order = [bt] + [x for x in s.car_numbers if x != bt]
            btr.append(t)
        bmodel = train_gbdt(btr)

        for s, d, ln, bt in te:
            st = model.strengths(s.X, s.car_numbers)
            pb = bmodel.strengths(s.X, s.car_numbers)
            if not st or not pb:
                continue
            npos = {c: p for c, (_l, p) in d.items()}
            mix, dists = branch_mixture(st, ln, pb)
            if not mix:
                continue                      # 3方式を同じ母集団で比べる
            dist = {"pl": all_trifecta_probs(st),
                    "himo": corrected_trifecta_probs(st, npos or None, MEN_PARAMS),
                    "mix": mix}
            win, yen = pay[s.race_id]
            wt = tuple(int(x) for x in win.split("-"))
            for k in names:
                pr = dist[k]
                top = [x for x, _ in sorted(pr.items(), key=lambda kv: -kv[1])[:10]]
                tri10[k][0] += int(wt in top); tri10[k][1] += 1
                for t in TARGETS:
                    buy = pick(pr, t)
                    h = int(wt in buy)
                    acc[(k, t)].append((STAKE * len(buy), yen if h else 0))
                    hits[(k, t)][0] += h; hits[(k, t)][1] += 1

    print(f"\n参考: tri10  " + " / ".join(
        f"{k} {tri10[k][0]/tri10[k][1]*100:.2f}%" for k in names))
    print(f"\n{'目標':>5}{'方式':>6}{'点数':>7}{'的中率':>8}{'回収率':>9}{'95%区間':>20}")
    res = {}
    for t in TARGETS:
        for k in names:
            rr = acc[(k, t)]
            pts = sum(a for a, _ in rr) / len(rr) / STAKE
            hit = hits[(k, t)][0] / hits[(k, t)][1] * 100
            p, lo, hi = boot(rr)
            res[(k, t)] = (p, lo, hi)
            print(f"{t*100:>4.0f}%{k:>6}{pts:>7.1f}{hit:>7.1f}%{p*100:>8.1f}%"
                  f"{f'[{lo*100:.1f}–{hi*100:.1f}%]':>20}")
        print()

    print("事前登録した判定（mix の回収率が pl より有意に高いか）:")
    ok_any = False
    for t in TARGETS:
        pm, lom, him = res[("mix", t)]
        pp, lop, hip = res[("pl", t)]
        sep = lom > hip                      # 区間が重ならない
        ok_any = ok_any or sep
        print(f"  目標{t*100:.0f}%: mix {pm*100:.1f}%[{lom*100:.1f}–{him*100:.1f}] vs "
              f"pl {pp*100:.1f}%[{lop*100:.1f}–{hip*100:.1f}] → "
              f"{'有意に高い' if sep else '区間が重なる＝差を主張できない'}")
    print(f"\n→ {'モデル改善はROIに伝わる' if ok_any else 'モデル改善はROIに伝わらない'}")
    print("※ tri10 は大きく違うのに回収率が動かないなら、精度への投資は回収率を生まない。")


if __name__ == "__main__":
    main()
