"""(c) 展開分岐の較正を walk-forward で out-of-sample 検証する。

**なぜ要るか**: 役割倍率(branch_stats_men.json)も mate_boost=4.0 も、モデルと同じ
25,000レースで測った **in-sample** の値。「実測に一致するよう合わせた」のだから
同じデータで一致するのは当たり前で、それは検証ではない。

**この検証の作り**（fold ごとに全部やり直す）:
  train fold で  ①着順モデル ②展開AI(B予測) ③役割倍率 ④mate_boost  を作り、
  test fold（未来）でだけ評価する。①②を全データ学習のままにすると
  「較正だけ分けた」ことになり検証にならないので、モデルも fold ごとに学習する。

**事前登録した採否基準（後から緩めない）**:
  主基準: test fold でのライン決着のズレ |予測−実測| が **5pt以内** を過半fold(3/5以上)
  副基準: test fold の三連単 log-loss が素のPLより改善（過半fold）
  外れたら**ダッシュボードに出さない**。in-sampleで合っているだけの数字を表示しない。

  PYTHONIOENCODING=utf-8 python scripts/validate_branch_calibration.py
"""
from __future__ import annotations

import argparse
import math
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
from src.model.development_branches import role_of, branch_trifecta, ROLES
from src.backtest.walkforward import fold_boundaries

MB_GRID = [1.5, 2.0, 2.5, 3.0, 3.5]
LB_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]

# 条件付き2着を見るときの役割は「**勝者**から見た相対位置」。主導権者基準ではない
# （買い目は「Aが勝ったとき誰が2着か」で組むので、基準は勝者側でなければ読み解けない）。
COND_ROLES = ["勝者の番手", "勝者と同ライン他", "他ライン先頭", "他ライン番手",
              "他ライン3番手+", "単騎"]


def _cond_role(car: int, winner: int, lines: list[list[int]]) -> str:
    lo = {c: i for i, mem in enumerate(lines) for c in mem}
    lw, lc = lo.get(winner), lo.get(car)
    if lc is None or lw is None:
        return "単騎"
    if lc == lw:
        mem = lines[lw]
        i = mem.index(winner)
        return "勝者の番手" if (i + 1 < len(mem) and mem[i + 1] == car) else "勝者と同ライン他"
    mem = lines[lc]
    if len(mem) == 1:
        return "単騎"
    i = mem.index(car)
    return "他ライン先頭" if i == 0 else ("他ライン番手" if i == 1 else "他ライン3番手+")


def _ctx(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    nb = defaultdict(dict)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        nb[rid][car] = (li, pi)
    pos, sbm = defaultdict(dict), defaultdict(dict)
    for rid, p, car, s in c.execute("SELECT race_id,position,car_number,sb FROM results"):
        pos[rid][p] = car
        sbm[rid][car] = s
    c.close()
    return nb, pos, sbm


def _lines(d: dict) -> list[list[int]]:
    mem = defaultdict(list)
    for car, (li, pi) in d.items():
        mem[li].append((pi, car))
    return [[c for _, c in sorted(v)] for _, v in sorted(mem.items())]


def _b_taker(rid, d, sbm):
    bs = [c for c in d if sbm[rid].get(c) and "B" in str(sbm[rid][c])]
    return bs[0] if len(bs) == 1 else None


def _fit_weights(rows, model) -> dict:
    """train fold で役割倍率を測る（実測シェア ÷ 素のPLのシェア）。"""
    act = [defaultdict(int) for _ in range(3)]
    mdl = [defaultdict(float) for _ in range(3)]
    for s, d, lines, b, order in rows:
        st = model.strengths(s.X, s.car_numbers)
        if not st:
            continue
        for k in range(3):
            if k < len(order) and order[k] in d:
                act[k][role_of(order[k], b, lines)] += 1
        for combo, p in all_trifecta_probs(st).items():
            for k, car in enumerate(combo):
                if car in d:
                    mdl[k][role_of(car, b, lines)] += p
    w = {}
    for k in range(3):
        ta = sum(act[k].values()) or 1
        tm = sum(mdl[k].values()) or 1.0
        w[str(k + 1)] = {r: (act[k].get(r, 0) / ta) / (mdl[k].get(r, 0.0) / tm)
                         if mdl[k].get(r, 0.0) / tm > 1e-9 else 1.0 for r in ROLES}
    return {"weights": w}


def _settle(dist, lines) -> float:
    lo = {c: i for i, mem in enumerate(lines) for c in mem}
    return sum(p for (a, b, _), p in dist.items()
               if lo.get(a) is not None and lo.get(a) == lo.get(b))


def _cond_kind(x, w, lines):
    lo = {c: i for i, m in enumerate(lines) for c in m}
    lw, lx = lo.get(w), lo.get(x)
    if lx is None or lw is None or lx != lw:
        return "other"
    mem = lines[lw]
    i = mem.index(w)
    return "mate" if (i + 1 < len(mem) and mem[i + 1] == x) else "same"


def _fit_boosts(rows, model, stats) -> tuple[float, float]:
    """train fold で **2つの実測シェア**（勝者の番手が2着 / 同ライン他が2着）に同時に合わせる。

    合計（ライン決着率）だけに合わせると配分を誤る。mate単独較正では合計が合うのに
    番手42.2%(実測32.2%) / 同ライン他13.4%(実測23.1%) と ±10pt ずれた。
    """
    cache = []
    a_m = a_s = 0
    for s, d, lines, b, order in rows:
        st = model.strengths(s.X, s.car_numbers)
        if not st or len(order) < 2:
            continue
        cache.append((st, b, lines, order[0]))
        k = _cond_kind(order[1], order[0], lines)
        a_m += int(k == "mate")
        a_s += int(k == "same")
    if not cache:
        return 0.0, 0.0
    n = len(cache)
    t_m, t_s = a_m / n, a_s / n
    best, bd = (0.0, 0.0), 1e9
    for mb in MB_GRID:
        for lb in LB_GRID:
            pm = ps = 0.0
            for st, b, lines, w in cache:
                dd = branch_trifecta(st, b, lines, stats, mate_boost=mb, line_boost=lb)
                sub = {}
                for (x1, x2, _x3), p in dd.items():
                    if x1 == w:
                        sub[x2] = sub.get(x2, 0.0) + p
                z = sum(sub.values())
                if z <= 0:
                    continue
                for x, p in sub.items():
                    k = _cond_kind(x, w, lines)
                    if k == "mate":
                        pm += p / z
                    elif k == "same":
                        ps += p / z
            err = abs(pm / n - t_m) + abs(ps / n - t_s)
            if err < bd:
                best, bd = (mb, lb), err
    return best


def _fit_mate_boost(rows, model, stats) -> float:
    """（旧）ライン決着の合計だけに合わせる較正。配分を誤るので _fit_boosts を使う。"""
    act = tot = 0
    cache = []
    for s, d, lines, b, order in rows:
        st = model.strengths(s.X, s.car_numbers)
        if not st or len(order) < 2:
            continue
        lo = {c: i for i, mem in enumerate(lines) for c in mem}
        if lo.get(order[0]) is None or lo.get(order[1]) is None:
            continue
        cache.append((st, b, lines))
        act += int(lo[order[0]] == lo[order[1]])
        tot += 1
    if not tot:
        return 0.0
    target = act / tot
    best, bd = 0.0, 1e9
    for mb in MB_GRID:
        pred = sum(_settle(branch_trifecta(st, b, ln, stats, mate_boost=mb), ln)
                   for st, b, ln in cache) / len(cache)
        if abs(pred - target) < bd:
            best, bd = mb, abs(pred - target)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="展開分岐の較正を out-of-sample 検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    nb, pos, sbm = _ctx(args.db)
    raw = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(raw, args.db, men_features())

    rows = []                     # (sample, narabi, lines, B取得車, 着順)
    for s in samples:
        d = nb.get(s.race_id)
        P = pos.get(s.race_id)
        if not d or not P:
            continue
        b = _b_taker(s.race_id, d, sbm)
        if b is None:
            continue
        order = [P.get(k) for k in (1, 2, 3)]
        if order[0] is None or order[1] is None:
            continue
        rows.append((s, d, _lines(d), b, [x for x in order if x is not None]))
    print(f"対象 {len(rows):,}レース（B一意・着順あり）")

    bounds = fold_boundaries(len(rows), n_folds=args.folds, warmup_frac=0.40,
                             window="expanding")
    print(f"\n{'fold':>5}{'n_test':>8}{'mate_boost':>11}{'実測':>8}{'素のPL':>9}"
          f"{'分岐':>8}{'PLのズレ':>10}{'分岐のズレ':>11}{'LL素PL':>9}{'LL分岐':>9}")
    d_settle, d_ll = [], []
    _fold_models = {}
    for fi, (a, b2, c) in enumerate(bounds):
        tr, te = rows[a:b2], rows[b2:c]
        model = train_gbdt([r[0] for r in tr])
        stats = _fit_weights(tr, model)
        mb, lb = _fit_boosts(tr, model, stats)
        stats["mate_boost"], stats["line_boost"] = mb, lb
        _fold_models[fi] = {"model": model, "stats": stats, "mb": mb, "lb": lb}

        act = n = 0
        s_pl = s_br = 0.0
        ll_pl = ll_br = 0.0
        for s, d, lines, bt, order in te:
            st = model.strengths(s.X, s.car_numbers)
            if not st:
                continue
            lo = {c2: i for i, mem in enumerate(lines) for c2 in mem}
            if lo.get(order[0]) is None or lo.get(order[1]) is None:
                continue
            # 主導権は**予測しない**。分岐の較正そのものを見るため真のBを条件にする。
            # （P(B)の不確実性を混ぜると較正の良し悪しが分離できない）
            dpl = all_trifecta_probs(st)
            dbr = branch_trifecta(st, bt, lines, stats, mate_boost=mb, line_boost=lb)
            s_pl += _settle(dpl, lines)
            s_br += _settle(dbr, lines)
            act += int(lo[order[0]] == lo[order[1]])
            n += 1
            if len(order) >= 3:
                k = tuple(order[:3])
                ll_pl -= math.log(max(dpl.get(k, 0.0), 1e-12))
                ll_br -= math.log(max(dbr.get(k, 0.0), 1e-12))
        if not n:
            continue
        A, P1, P2 = act / n * 100, s_pl / n * 100, s_br / n * 100
        d_settle.append(abs(P2 - A))
        d_ll.append(ll_br / n - ll_pl / n)
        print(f"{fi:>5}{n:>8}{str(mb)+chr(47)+str(lb):>11}{A:>7.1f}%{P1:>8.1f}%{P2:>7.1f}%"
              f"{P1-A:>+10.1f}{P2-A:>+11.1f}{ll_pl/n:>9.3f}{ll_br/n:>9.3f}")

    # --- 条件付き2着/3着の較正（買い目の土台になるので必ず見る）---
    # 「展開Xで、Aが勝ったときの2着・3着」は P(2着|B, 1着) そのもの。買い目をここから
    # 組む以上、周辺のライン決着だけでなく**この条件付き分布**が当たっている必要がある。
    print("\n=== 条件付き2着の較正（真のB・真の1着を条件・test foldのみ）===")
    print(f"{'役割':>16}{'実測':>9}{'素のPL':>9}{'分岐':>9}{'PL誤差':>9}{'分岐誤差':>10}")
    ta = {r: 0 for r in COND_ROLES}
    tp = {r: 0.0 for r in COND_ROLES}
    tb = {r: 0.0 for r in COND_ROLES}
    ncond = 0
    for fi, (a, b2, c) in enumerate(bounds):
        tr, te = rows[a:b2], rows[b2:c]
        model = _fold_models[fi]["model"]
        stats = _fold_models[fi]["stats"]
        mb = _fold_models[fi]["mb"]
        lb = _fold_models[fi]["lb"]
        for s2, d, lines, bt, order in te:
            if len(order) < 2:
                continue
            st = model.strengths(s2.X, s2.car_numbers)
            if not st:
                continue
            w = order[0]
            if w not in st:
                continue
            dpl = all_trifecta_probs(st)
            dbr = branch_trifecta(st, bt, lines, stats, mate_boost=mb, line_boost=lb)
            # 1着=w で条件付けた2着分布
            def cond2(dd):
                sub = {}
                for (x1, x2, _x3), p in dd.items():
                    if x1 == w:
                        sub[x2] = sub.get(x2, 0.0) + p
                z = sum(sub.values())
                return {k: v / z for k, v in sub.items()} if z > 0 else {}
            cpl, cbr = cond2(dpl), cond2(dbr)
            if not cbr:
                continue
            ncond += 1
            ta[_cond_role(order[1], w, lines)] += 1
            for x, p in cpl.items():
                tp[_cond_role(x, w, lines)] += p
            for x, p in cbr.items():
                tb[_cond_role(x, w, lines)] += p
    if ncond:
        for r in COND_ROLES:
            A, P1, P2 = ta[r] / ncond * 100, tp[r] / ncond * 100, tb[r] / ncond * 100
            print(f"{r:>16}{A:>8.1f}%{P1:>8.1f}%{P2:>8.1f}%{P1-A:>+9.1f}{P2-A:>+10.1f}")
        mae_pl = sum(abs(tp[r] - ta[r]) for r in COND_ROLES) / ncond * 100 / len(COND_ROLES)
        mae_br = sum(abs(tb[r] - ta[r]) for r in COND_ROLES) / ncond * 100 / len(COND_ROLES)
        print(f"  平均絶対誤差: 素のPL {mae_pl:.2f}pt → 分岐 {mae_br:.2f}pt  (n={ncond:,})")

    n_ok = sum(1 for d in d_settle if d <= 5.0)
    n_ll = sum(1 for d in d_ll if d < 0)
    print(f"\n平均: 分岐のズレ {sum(d_settle)/len(d_settle):.1f}pt / "
          f"log-loss差 {sum(d_ll)/len(d_ll):+.4f}")
    print("\n事前基準の判定:")
    print(f"  主基準（ライン決着のズレ5pt以内が3/5fold以上）: {n_ok}/{len(d_settle)} → "
          f"{'充足' if n_ok >= 3 else '不充足'}")
    print(f"  副基準（三連単log-lossが素のPLより改善）: {n_ll}/{len(d_ll)} → "
          f"{'充足' if n_ll >= 3 else '不充足'}")
    print(f"\n→ {'採用（ダッシュボードに出す）' if n_ok >= 3 and n_ll >= 3 else '不採用（表示しない）'}")


if __name__ == "__main__":
    main()
