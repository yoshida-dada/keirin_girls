"""fav_fade（◎が1着を外したときの失速）を実測に合わせて較正する。

**なぜ要るか**: 買い目の型別検証で「◎が飛ぶ」ケースを 15.9%（実測23.2%）と **7.3pt過小評価**
していた。◎頭の誤差は+0.6ptとほぼ完璧なので1着確率は合っており、ずれているのは
「◎が1着を外したあとどこまで落ちるか」。PLは強い選手を2・3着に残しすぎる。

実測（男子25,156R・◎は競走得点1位で近似）:
  ◎が1着を外した 62.9%。そのうち **2着34.7% / 3着21.3% / 3着圏外44.0%**。

目標は ◎2着 / ◎3着 の出現率（◎頭は fav_fade で動かないので合わせる必要がない）。
**手で決めない。** 候補値を実測と突き合わせて選ぶ。

  PYTHONIOENCODING=utf-8 python scripts/calibrate_fav_fade.py --emit
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import men_features, load_for
from src.model.development_branches import branch_trifecta, STATS_PATH

GRID = [1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3]


def main() -> None:
    ap = argparse.ArgumentParser(description="fav_fade の較正")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()

    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    nb, sbm, pos = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        nb[rid][car] = (li, pi)
    for rid, p, car, s in c.execute("SELECT race_id,position,car_number,sb FROM results"):
        pos[rid][p] = car
        sbm[rid][car] = s
    c.close()

    model, _e, _l = load_for(False)
    raw = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(raw, args.db, men_features())
    if args.limit:
        samples = samples[-args.limit:]

    rows = []
    a1 = a2 = a3 = 0
    for s in samples:
        d, P = nb.get(s.race_id), pos.get(s.race_id)
        if not d or not P or 1 not in P or 2 not in P or 3 not in P:
            continue
        bs = [x for x in d if sbm[s.race_id].get(x) and "B" in str(sbm[s.race_id][x])]
        if len(bs) != 1:
            continue
        st = model.strengths(s.X, s.car_numbers)
        if not st:
            continue
        mem = defaultdict(list)
        for car, (li, pi) in d.items():
            mem[li].append((pi, car))
        lines = [[x for _, x in sorted(v)] for _, v in sorted(mem.items())]
        fav = max(st, key=st.get)
        rows.append((st, bs[0], lines, fav))
        a1 += int(P[1] == fav)
        a2 += int(P[2] == fav)
        a3 += int(P[3] == fav)

    n = len(rows)
    t1, t2, t3 = a1 / n, a2 / n, a3 / n
    print(f"n={n:,}  実測: ◎頭 {t1*100:.1f}% / ◎2着 {t2*100:.1f}% / ◎3着 {t3*100:.1f}% "
          f"/ ◎抜き {(1-t1-t2-t3)*100:.1f}%\n")

    best = None
    print(f"{'fav_fade':>9}{'◎頭':>8}{'◎2着':>8}{'◎3着':>8}{'◎抜き':>9}{'誤差和':>9}")
    for ff in GRID:
        p1 = p2 = p3 = 0.0
        for st, b, lines, fav in rows:
            dd = branch_trifecta(st, b, lines, fav_fade=ff)
            p1 += sum(p for k, p in dd.items() if k[0] == fav)
            p2 += sum(p for k, p in dd.items() if k[1] == fav)
            p3 += sum(p for k, p in dd.items() if k[2] == fav)
        p1, p2, p3 = p1 / n, p2 / n, p3 / n
        err = abs(p2 - t2) + abs(p3 - t3)
        print(f"{ff:>9.2f}{p1*100:>7.1f}%{p2*100:>7.1f}%{p3*100:>7.1f}%"
              f"{(1-p1-p2-p3)*100:>8.1f}%{err*100:>8.2f}")
        if best is None or err < best[1]:
            best = (ff, err, p1, p2, p3)
    print(f"\n→ 最良 fav_fade={best[0]} （誤差和 {best[1]*100:.2f}pt）")
    if best[0] in (GRID[0], GRID[-1]):
        print("  ⚠ 範囲の端を引いた。GRIDを広げて再実行すること")
    if args.emit:
        st = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        st["fav_fade"] = best[0]
        st["fav_fade_note"] = (
            "◎が1着を外したときの2着/3着重みの倍率。PLは強い選手を2・3着に残しすぎるため、"
            "買い目の型別検証で「◎が飛ぶ」を15.9%(実測23.2%)と7.3pt過小評価していた。"
            f"実測{n:,}Rの◎2着/◎3着の出現率へ較正して {best[0]}"
            f"（◎2着{best[3]*100:.1f}% / ◎3着{best[4]*100:.1f}%）。手で決めた値ではない。")
        STATS_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"更新: {STATS_PATH.name}  fav_fade={best[0]}")


if __name__ == "__main__":
    main()
