"""市場ブレンドで**予測精度**がどれだけ上がる余地があるか（男子7車）。

収益方向のブレンドは検証済みで効かない（keirin-roi-strategy 2026-07-16: エッジ比を
上げるほどROIが下がる＝市場との乖離はエッジでなくノイズ）。ここで測るのは**別の問い**:
表示している確率そのものの質が、市場を混ぜるとどれだけ良くなるか。

**なぜ今これを測るか**: 4.21.2 で「EV順に選ぶと予測的中率10%に対し実測3.7%＝
モデル確率が高配当帯で2.7倍過大」と定量化された。裾の較正が悪いことが分かっている以上、
市場（裾の値付けは市場の方が正確と 4.21 で判明）を混ぜれば直る見込みがある。

**重要な制約**: 確定オッズを使うので、これは**締切間近の表示にしか適用できない**。
発売前の予測には市場が存在しないので混ぜられない。よって「モデルを置き換える」話ではなく
「締切間近に表示を差し替えられるか」の話。学習には一切使わない（リーク防止）。

α はプロジェクトの慣行に合わせ **モデル側の重み**（α=1 で純モデル、α=0 で純市場）。
α=0 の行が**到達可能な上限**＝市場そのものの精度で、これが「向上の余地」の天井。

**事前登録した採否基準（後から緩めない）**:
  採用するなら 主基準: 万車券率の ECE が α=1 より **半分以下**、かつ tri10 が悪化しない
              副基準: 5foldのうち4fold以上で同じ向き
  ここでは余地の測定が目的だが、基準を先に置いておかないと後から都合よく読める。

  PYTHONIOENCODING=utf-8 python scripts/validate_blend_accuracy.py
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
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.upset import threshold_for
from src.ev.market import implied_trifecta_probs, blend_loglinear
from src.backtest.walkforward import fold_boundaries

ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0]


def _ece(pred, y, bins=10):
    if not pred:
        return 0.0
    idx = sorted(range(len(pred)), key=lambda i: pred[i])
    n, tot = len(pred), 0.0
    for b in range(bins):
        part = idx[b * n // bins:(b + 1) * n // bins]
        if not part:
            continue
        pm = sum(pred[i] for i in part) / len(part)
        ym = sum(y[i] for i in part) / len(part)
        tot += len(part) * abs(pm - ym)
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser(description="市場ブレンドの予測精度への効果")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    odds_all = defaultdict(dict)
    for rid, combo, o in c.execute("SELECT race_id,combo,odds FROM odds_final_trifecta"):
        a, b, cc = (int(x) for x in combo.split("-"))
        odds_all[rid][(a, b, cc)] = o
    pay = {r: (combo, p) for r, combo, p in
           c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    nar = defaultdict(dict)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi"
            " WHERE line_id IS NOT NULL"):
        nar[rid][car] = (li, pi)
    c.close()

    thr = threshold_for(False, 7)
    raw = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    smp = augment_samples(raw, args.db, men_features())
    print(f"サンプル {len(smp):,}（男子7車）  万車券率のしきい {thr}")

    # acc[alpha] = {...}
    acc = {a: {"tri10": 0, "top1": 0, "ll": 0.0, "n": 0,
               "up_p": [], "up_y": [], "ev_p": 0.0, "ev_n": 0, "ev_hit": 0}
           for a in ALPHAS}
    facc = {a: defaultdict(lambda: [0, 0, 0.0]) for a in ALPHAS}   # fold -> [tri10, n, ece用は別]
    fup = {a: defaultdict(lambda: ([], [])) for a in ALPHAS}

    for fi, (a0, b0, c0) in enumerate(fold_boundaries(len(smp), n_folds=args.folds,
                                                       warmup_frac=0.40, window="expanding")):
        model = train_gbdt(smp[a0:b0])
        for s in smp[b0:c0]:
            od = odds_all.get(s.race_id)
            if not od or s.race_id not in pay:
                continue
            st = model.strengths(s.X, s.car_numbers)
            if not st:
                continue
            npos = {cc: p for cc, (_l, p) in (nar.get(s.race_id) or {}).items()}
            base = corrected_trifecta_probs(st, npos or None, MEN_PARAMS)
            imp = implied_trifecta_probs(od)
            truth = tuple(s.order[:3])
            for al in ALPHAS:
                pr = base if al >= 1.0 else blend_loglinear(base, imp, al)
                z = sum(pr.values())
                if z <= 0:
                    continue
                pr = {k: v / z for k, v in pr.items()}
                A = acc[al]
                A["n"] += 1
                top = sorted(pr.items(), key=lambda kv: -kv[1])[:10]
                hit10 = int(truth in [k for k, _ in top])
                A["tri10"] += hit10
                facc[al][fi][0] += hit10
                facc[al][fi][1] += 1
                # 1着確率は三連単分布から周辺化する（ブレンド後も一貫させるため）
                win = defaultdict(float)
                for (x, _y, _z2), p in pr.items():
                    win[x] += p
                A["top1"] += int(max(win, key=win.get) == truth[0])
                A["ll"] += -math.log(max(pr.get(truth, 1e-12), 1e-12))
                # 万車券率の較正
                up = sum(p for p in pr.values() if p <= thr)
                A["up_p"].append(up)
                A["up_y"].append(int(pay[s.race_id][1] >= 10000))
                fup[al][fi][0].append(up)
                fup[al][fi][1].append(int(pay[s.race_id][1] >= 10000))
                # 裾の壊れ具合: EV順に累積10%まで買ったときの予測と実測
                seq = sorted(pr.items(), key=lambda kv: -(kv[1] * od.get(kv[0], 0.0)))
                cum, buy = 0.0, []
                for k, p in seq:
                    buy.append(k); cum += p
                    if cum >= 0.10:
                        break
                A["ev_p"] += cum
                A["ev_n"] += 1
                A["ev_hit"] += int(truth in buy)

    print(f"\n{'α(モデル重み)':>13}{'tri10':>9}{'top1':>8}{'logloss':>10}"
          f"{'万車券ECE':>11}{'EV順10%の実測':>15}")
    base_ece = None
    for al in ALPHAS:
        A = acc[al]
        if not A["n"]:
            continue
        e = _ece(A["up_p"], A["up_y"])
        if al >= 1.0:
            base_ece = e
        lbl = f"{al:.1f}" + ("（純モデル）" if al >= 1.0 else ("（純市場）" if al == 0 else ""))
        print(f"{lbl:>13}{A['tri10']/A['n']*100:>8.2f}%{A['top1']/A['n']*100:>7.2f}%"
              f"{A['ll']/A['n']:>10.4f}{e:>11.4f}"
              f"{A['ev_hit']/A['ev_n']*100:>14.1f}%")

    print(f"\n※ EV順10%は「予測的中率10%まで買ったときの実測的中率」。"
          f"純モデルで3.7%なら裾が2.7倍過大の意味。10%に近いほど裾が直っている。")
    print("\n事前基準（採用するなら）: 万車券ECEが純モデルの半分以下 かつ tri10 が悪化しない")
    for al in ALPHAS:
        if al >= 1.0 or not acc[al]["n"]:
            continue
        e = _ece(acc[al]["up_p"], acc[al]["up_y"])
        t = acc[al]["tri10"] / acc[al]["n"]
        t0 = acc[1.0]["tri10"] / acc[1.0]["n"]
        wins = sum(1 for fi in facc[al]
                   if facc[al][fi][1] and facc[1.0][fi][1]
                   and facc[al][fi][0] / facc[al][fi][1] >= facc[1.0][fi][0] / facc[1.0][fi][1])
        ok = e <= base_ece / 2 and t >= t0
        print(f"  α={al:.1f}: ECE {e:.4f}(基準{base_ece/2:.4f}以下) / "
              f"tri10 {t*100:.2f}%(基準{t0*100:.2f}%以上) / tri10非悪化 {wins}/{args.folds}fold"
              f"  → {'充足' if ok else '不充足'}")
    print("\n※ 確定オッズを使うので**締切間近の表示にしか適用できない**。"
          "発売前の予測には市場が無い。学習には一切使っていない。")


if __name__ == "__main__":
    main()
