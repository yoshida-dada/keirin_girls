"""ライン特徴の純増を walk-forward で検証（男子・基準 vs +ライン8列）。

基準は「拡張20特徴 + rel_elo」。ガールズ用にチューニングされた展開/並び特徴は
男子では意味が変わるので入れず、**ライン特徴の寄与だけを切り出す**。

**事前登録した採否基準（後から緩めない）**:
  主基準: 全体 tri10 が過半foldで改善、かつ 全体 top1 の平均が悪化しない
  副基準: 全体 top1 が過半foldで改善
  外れたら打ち切り。

  PYTHONIOENCODING=utf-8 python scripts/validate_line_features.py --db data/keirin_men.sqlite
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

from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate
from src.model.feature_augment import augment_samples
from src.model.plackett_luce import all_trifecta_probs
from src.features.line_features import LINE_KEYS, line_columns, class_level
from src.backtest.walkforward import fold_boundaries


def _ctx(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    line_of = defaultdict(dict)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        line_of[rid][car] = (li, pi)
    scores = defaultdict(dict)
    cls = defaultdict(list)
    for rid, car, sc, cr in c.execute(
            "SELECT race_id,car_number,racing_score,class_rank FROM entries"):
        if sc:
            scores[rid][car] = sc
        if cr:
            cls[rid].append(cr)
    c.close()
    return line_of, scores, cls


def _tri10(model, test) -> float:
    if not test:
        return 0.0
    hit = 0
    for s in test:
        ranked = sorted(all_trifecta_probs(model.strengths(s.X, s.car_numbers)).items(),
                        key=lambda kv: -kv[1])[:10]
        hit += int(tuple(s.order[:3]) in [k for k, _ in ranked])
    return hit / len(test)


def _add_line(samples, line_of, scores, cls):
    out = []
    for s in samples:
        cars = list(s.car_numbers)
        cols = line_columns(cars, line_of.get(s.race_id, {}), scores.get(s.race_id, {}),
                            class_level(cls.get(s.race_id, [])))
        s2 = copy.copy(s)
        s2.X = np.hstack([s.X, np.array([cols[c] for c in cars], dtype=float)])
        s2.feature_names = list(s.feature_names) + list(LINE_KEYS)
        out.append(s2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ライン特徴の検証（男子）")
    ap.add_argument("--db", default="data/keirin_men.sqlite")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--field", default="7", help="車立て: 7 / 7,9 / all")
    args = ap.parse_args()

    fs = None if args.field == "all" else [int(x) for x in args.field.split(",")]
    line_of, scores, cls = _ctx(args.db)
    base = load_samples(args.db, field_size=fs, features=PL_FEATURES_FULL)
    s0 = augment_samples(base, args.db, list(PL_FEATURES_FULL) + ["rel_elo"])
    s1 = _add_line(s0, line_of, scores, cls)
    print(f"男子 {len(s0)}R（車立て {args.field}）  基準{s0[0].X.shape[1]}特徴 "
          f"→ ライン込み{s1[0].X.shape[1]}特徴")

    # 自己チェック: ライン列が実際に非ゼロか（0のままだと「効果なし」と誤読する）
    nz = sum(1 for s in s1 if np.abs(s.X[:, -len(LINE_KEYS):]).sum() > 0)
    print(f"  ライン列が非ゼロなレース: {nz}/{len(s1)}")
    if nz == 0:
        raise SystemExit("★ ライン特徴が全て0。並び予想の紐付けに失敗している。")

    bounds = fold_boundaries(len(base), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"\nwalk-forward {len(bounds)}fold\n")
    print(f"{'fold':>4}{'testR':>8}{'  top1 基準→ライン':>22}{'  tri10 基準→ライン':>23}{'   ece 基準→ライン':>22}")
    agg = []
    for i, (a, b, c) in enumerate(bounds):
        m0, m1 = train_gbdt(s0[a:b]), train_gbdt(s1[a:b])
        t0, t1 = s0[b:c], s1[b:c]
        e0, e1 = evaluate(m0.strengths, t0), evaluate(m1.strengths, t1)
        r0, r1 = _tri10(m0, t0), _tri10(m1, t1)
        agg.append({"top1": e1["top1_acc"] - e0["top1_acc"], "tri10": r1 - r0,
                    "ece": e1["ece"] - e0["ece"], "ll": e1["logloss"] - e0["logloss"]})
        print(f"{i:>4}{len(t0):>8}{e0['top1_acc']*100:>13.1f}→{e1['top1_acc']*100:.1f}%"
              f"{r0*100:>14.1f}→{r1*100:.1f}%{e0['ece']:>13.4f}→{e1['ece']:.4f}")

    n = len(agg)
    av = lambda k: sum(r[k] for r in agg) / n
    wins = lambda k, low=False: sum(1 for r in agg if (r[k] < 0 if low else r[k] > 0))
    maj = n // 2 + 1
    print(f"\n【ライン特徴の純増】{n}fold平均・+勝ちfold数")
    print(f"  top1    {av('top1')*100:+.2f}pt  勝ち {wins('top1')}/{n}   ←副基準")
    print(f"  tri10   {av('tri10')*100:+.2f}pt  勝ち {wins('tri10')}/{n}   ←主基準")
    print(f"  logloss {av('ll'):+.4f}   改善 {wins('ll', True)}/{n}")
    print(f"  ece     {av('ece'):+.5f}  改善 {wins('ece', True)}/{n}")
    ok_main = wins("tri10") >= maj and av("top1") >= 0
    ok_sub = wins("top1") >= maj
    print(f"\n  主基準 {'充足' if ok_main else '未充足'} / 副基準 {'充足' if ok_sub else '未充足'}"
          f"  → {'採用検討' if (ok_main and ok_sub) else ('主基準のみ＝要追加確認' if ok_main else '不採用')}")


if __name__ == "__main__":
    main()
