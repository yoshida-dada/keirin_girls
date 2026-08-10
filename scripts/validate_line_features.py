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
from src.features.line_features import (LINE_KEYS, LEG_KEYS, line_columns,
                                        class_level, leg_aggr)
from src.features.tactics_features import TACTIC_NAMES
from src.backtest.walkforward import fold_boundaries


def _ctx(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    line_of = defaultdict(dict)
    legs = defaultdict(dict)
    for rid, car, li, pi, leg in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line,leg FROM narabi"):
        if li is not None:
            line_of[rid][car] = (li, pi)
        if leg:
            legs[rid][car] = leg
    scores = defaultdict(dict)
    cls = defaultdict(list)
    for rid, car, sc, cr in c.execute(
            "SELECT race_id,car_number,racing_score,class_rank FROM entries"):
        if sc:
            scores[rid][car] = sc
        if cr:
            cls[rid].append(cr)
    c.close()
    return line_of, scores, cls, legs


def _tri10(model, test) -> float:
    if not test:
        return 0.0
    hit = 0
    for s in test:
        ranked = sorted(all_trifecta_probs(model.strengths(s.X, s.car_numbers)).items(),
                        key=lambda kv: -kv[1])[:10]
        hit += int(tuple(s.order[:3]) in [k for k, _ in ranked])
    return hit / len(test)


def _add_line(samples, line_of, scores, cls, legs=None):
    """ライン8列（+ legs を渡せば脚質前がかり度1列）を足す。"""
    out = []
    for s in samples:
        cars = list(s.car_numbers)
        cols = line_columns(cars, line_of.get(s.race_id, {}), scores.get(s.race_id, {}),
                            class_level(cls.get(s.race_id, [])))
        mat = [list(cols[c]) for c in cars]
        names = list(LINE_KEYS)
        if legs is not None:
            lg = legs.get(s.race_id, {})
            for row, c in zip(mat, cars):
                row.append(leg_aggr(lg.get(c)))
            names = names + list(LEG_KEYS)
        s2 = copy.copy(s)
        s2.X = np.hstack([s.X, np.array(mat, dtype=float)])
        s2.feature_names = list(s.feature_names) + names
        out.append(s2)
    return out


def _add_tactics(samples, db: str, names: list[str]):
    """展開特徴（選手履歴由来）を足す。augment_samples と同じ経路を使う。"""
    return augment_samples(samples, db, list(names))


def main() -> None:
    ap = argparse.ArgumentParser(description="ライン特徴の検証（男子）")
    ap.add_argument("--db", default="data/keirin_men.sqlite")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--field", default="7", help="車立て: 7 / 7,9 / all")
    ap.add_argument("--tactics", action="store_true", help="展開10列も足した案Dを含める")
    args = ap.parse_args()

    fs = None if args.field == "all" else [int(x) for x in args.field.split(",")]
    line_of, scores, cls, legs = _ctx(args.db)
    base = load_samples(args.db, field_size=fs, features=PL_FEATURES_FULL)
    s0 = augment_samples(base, args.db, list(PL_FEATURES_FULL) + ["rel_elo"])

    # 段階的に足して、どの塊がどれだけ効くかを切り分ける
    arms = [("A 基準(拡張20+Elo)", s0)]
    s1 = _add_line(s0, line_of, scores, cls)
    arms.append(("B +ライン8列", s1))
    s2 = _add_line(s0, line_of, scores, cls, legs=legs)
    arms.append(("C +ライン8+脚質1", s2))
    if args.tactics:
        tac = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
        st = _add_tactics(base, args.db, tac)
        arms.append(("D +展開10(脚質込)", _add_line(st, line_of, scores, cls, legs=legs)))
        # 脚質はライン位置に吸収され効かなかったので、それを外した構成も見る
        arms.append(("E ライン8+展開10", _add_line(st, line_of, scores, cls)))
    for lbl, ss in arms:
        print(f"  {lbl:<22} {ss[0].X.shape[1]:>3}特徴")

    nz = sum(1 for s in s1 if np.abs(s.X[:, -len(LINE_KEYS):]).sum() > 0)
    print(f"\nライン列が非ゼロなレース: {nz}/{len(s1)}")
    if nz == 0:
        raise SystemExit("★ ライン特徴が全て0。並び予想の紐付けに失敗している。")

    bounds = fold_boundaries(len(base), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"男子 {len(base)}R（車立て {args.field}）  walk-forward {len(bounds)}fold\n")

    res = {lbl: [] for lbl, _ in arms}
    for i, (a, b, c) in enumerate(bounds):
        for lbl, ss in arms:
            m = train_gbdt(ss[a:b])
            te = ss[b:c]
            e = evaluate(m.strengths, te)
            res[lbl].append({"top1": e["top1_acc"], "tri10": _tri10(m, te),
                             "ece": e["ece"], "ll": e["logloss"]})
        print(f"  fold{i} 完了")

    def avg(lbl, k):
        return sum(r[k] for r in res[lbl]) / len(res[lbl])

    print(f"\n{'案':<22}{'top1':>9}{'tri10':>9}{'logloss':>10}{'ece':>10}")
    for lbl, _ in arms:
        print(f"{lbl:<22}{avg(lbl,'top1')*100:>8.2f}%{avg(lbl,'tri10')*100:>8.2f}%"
              f"{avg(lbl,'ll'):>10.4f}{avg(lbl,'ece'):>10.5f}")

    maj = len(bounds) // 2 + 1

    def compare(prev: str, cur: str) -> None:
        d = {k: avg(cur, k) - avg(prev, k) for k in ("top1", "tri10", "ll", "ece")}
        w = {k: sum(1 for x, y in zip(res[cur], res[prev]) if x[k] > y[k])
             for k in ("top1", "tri10")}
        wl = sum(1 for x, y in zip(res[cur], res[prev]) if x["ll"] < y["ll"])
        ok = w["tri10"] >= maj and d["top1"] >= 0
        print(f"  {cur} − {prev}")
        print(f"    top1 {d['top1']*100:+.2f}pt({w['top1']}/{len(bounds)})  "
              f"tri10 {d['tri10']*100:+.2f}pt({w['tri10']}/{len(bounds)})  "
              f"logloss {d['ll']:+.4f}(改善{wl}/{len(bounds)})  ece {d['ece']:+.5f}")
        print(f"    → 主基準 {'充足' if ok else '未充足'}"
              f"（tri10 {w['tri10']}/{len(bounds)} >= {maj} かつ top1平均>=0）")


    names = [lbl for lbl, _ in arms]
    if "E ライン8+展開10" in names:
        print(f"\n【採用候補を基準Bと直接比較】{len(bounds)}fold平均")
        for cand in ("D +展開10(脚質込)", "E ライン8+展開10"):
            if cand in names:
                compare("B +ライン8列", cand)

    print(f"\n【直前の案からの純増】{len(bounds)}fold平均・+勝ちfold数")
    for j in range(1, len(arms)):
        prev, cur = arms[j - 1][0], arms[j][0]
        compare(prev, cur)



if __name__ == "__main__":
    main()


