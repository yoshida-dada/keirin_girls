"""同時確率の作り方3種を、ライン決着率の較正で比べる（男子7車）。

**背景**: 単勝確率→同時確率の変換は既に実装済み（`all_trifecta_probs` が全210通りを
厳密列挙）。Plackett-Luce の逐次選択は強さを Σ=1 に正規化すれば **Harville の式そのもの**
なので、「PLかHarvilleか」という選択肢は存在しない。210通りなら厳密列挙が常に
モンテカルロより良い（サンプリング誤差ゼロ）。

**本当の問題は独立性の崩れ**。競輪では「ライン連携」がそれで、素のPLはライン決着
（1-2着が同一ライン）を大幅に過小評価する。ここで比べるのは:

  pl    素のPL（＝Harville）
  himo  紐補正（2着/3着の重みを平坦化＋◎の番手を加点）。**着順ごとの周辺重み**の調整
  mix   展開分岐の混合分布 Σ P(B=b)·P(順位|B=b)。主導権で条件付けることで
        **同時共起そのもの**を作る。周辺重みの調整では作れない構造

測る指標:
  ライン決着率の較正   … 予測 vs 実測。ここが本題
  三連単 top10 的中     … 順位付けの質（同時確率を良くしても順位が落ちたら意味がない）
  logloss              … 実際に起きた目の確率

**事前登録した採否基準（後から緩めない）**:
  主基準: ライン決着率の |予測−実測| が pl より **半分以下**
  副基準: tri10 が pl より悪化しない
  mix が両方を満たせば「分岐混合を本表示の三連単確率に採用する」検討に進む。
  満たさなければ現状（本表示は himo 補正）を維持する。

  PYTHONIOENCODING=utf-8 python scripts/validate_joint.py
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
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.development_branches import branch_trifecta
from src.backtest.walkforward import fold_boundaries


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
    c.close()
    return nb, pos, sbm


def _lines(d):
    mem = defaultdict(list)
    for car, (li, pi) in d.items():
        mem[li].append((pi, car))
    return [[x for _, x in sorted(v)] for _, v in sorted(mem.items())]


def settle_prob(pr: dict, line_of: dict) -> float:
    """1着と2着が同一ラインになる確率（分布から積み上げ）。"""
    return sum(p for (a, b, _c), p in pr.items()
               if line_of.get(a) is not None and line_of.get(a) == line_of.get(b))


def main() -> None:
    ap = argparse.ArgumentParser(description="同時確率の作り方の比較")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    nb, pos, sbm = _ctx(args.db)
    raw = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    smp = augment_samples(raw, args.db, men_features())

    rows = []
    for s in smp:
        d, P = nb.get(s.race_id), pos.get(s.race_id)
        if not d or not P or 1 not in P or 2 not in P or 3 not in P:
            continue
        bs = [x for x in d if sbm[s.race_id].get(x) and "B" in str(sbm[s.race_id][x])]
        if len(bs) != 1:
            continue
        rows.append((s, d, _lines(d), bs[0], (P[1], P[2], P[3])))
    print(f"対象 {len(rows):,}レース（男子7車・B一意・3着確定）")

    names = ["pl", "himo", "mix"]
    acc = {k: {"settle": 0.0, "tri10": 0, "ll": 0.0, "n": 0} for k in names}
    actual_settle = 0
    for fi, (a0, b0, c0) in enumerate(fold_boundaries(len(rows), n_folds=args.folds,
                                                       warmup_frac=0.40, window="expanding")):
        tr, te = rows[a0:b0], rows[b0:c0]
        model = train_gbdt([r[0] for r in tr])
        btr = []
        for s, d, ln, bt, _o in tr:
            t = type(s)(**{**s.__dict__})
            t.order = [bt] + [x for x in s.car_numbers if x != bt]
            btr.append(t)
        bmodel = train_gbdt(btr)

        for s, d, ln, bt, order in te:
            st = model.strengths(s.X, s.car_numbers)
            pb = bmodel.strengths(s.X, s.car_numbers)
            if not st or not pb:
                continue
            line_of = {c: li for li, mem in enumerate(ln) for c in mem}
            npos = {c: p for c, (_l, p) in d.items()}
            dist = {}
            dist["pl"] = all_trifecta_probs(st)
            dist["himo"] = corrected_trifecta_probs(st, npos or None, MEN_PARAMS)
            mix = defaultdict(float)
            for b, p in sorted(pb.items(), key=lambda kv: -kv[1])[:3]:
                if p < 0.05:
                    continue
                for k, v in branch_trifecta(st, b, ln).items():
                    mix[k] += p * v
            z = sum(mix.values())
            dist["mix"] = {k: v / z for k, v in mix.items()} if z > 0 else {}
            actual_settle += int(line_of.get(order[0]) is not None
                                 and line_of.get(order[0]) == line_of.get(order[1]))
            truth = tuple(order)
            for k in names:
                pr = dist[k]
                if not pr:
                    continue
                A = acc[k]
                A["n"] += 1
                A["settle"] += settle_prob(pr, line_of)
                top = sorted(pr.items(), key=lambda kv: -kv[1])[:10]
                A["tri10"] += int(truth in [x for x, _ in top])
                A["ll"] += -math.log(max(pr.get(truth, 1e-12), 1e-12))

    n0 = acc["pl"]["n"]
    act = actual_settle / n0 * 100
    print(f"\n実測のライン決着率: {act:.2f}%  (n={n0:,})\n")
    print(f"{'方式':>6}{'ライン決着(予測)':>18}{'ズレ':>8}{'tri10':>9}{'logloss':>10}")
    gaps = {}
    for k in names:
        A = acc[k]
        sp = A["settle"] / A["n"] * 100
        gaps[k] = abs(sp - act)
        print(f"{k:>6}{sp:>17.2f}%{sp-act:>+8.2f}{A['tri10']/A['n']*100:>8.2f}%"
              f"{A['ll']/A['n']:>10.4f}")

    print("\n事前基準の判定（mix について）:")
    ok1 = gaps["mix"] <= gaps["pl"] / 2
    ok2 = acc["mix"]["tri10"] / acc["mix"]["n"] >= acc["pl"]["tri10"] / acc["pl"]["n"]
    print(f"  主基準（ライン決着のズレが pl の半分以下）: {gaps['mix']:.2f}pt "
          f"<= {gaps['pl']/2:.2f}pt → {'充足' if ok1 else '不充足'}")
    print(f"  副基準（tri10 が pl より悪化しない）: "
          f"{acc['mix']['tri10']/acc['mix']['n']*100:.2f}% vs "
          f"{acc['pl']['tri10']/acc['pl']['n']*100:.2f}% → {'充足' if ok2 else '不充足'}")
    print(f"\n→ {'分岐混合を本表示に採用する検討へ' if ok1 and ok2 else '現状維持（本表示は himo 補正）'}")
    print("\n※ 同時確率の較正が良くなっても収益エッジにはならない（回収率は別途検証済みで"
          "全手法100%未満）。価値は表示している確率の質。")


if __name__ == "__main__":
    main()
