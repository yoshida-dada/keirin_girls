"""mate_boost（1着の番手に掛ける2着重み）を実測のライン決着率に合わせて較正する。

**手で決めない。** 候補値を実測と突き合わせて選ぶ。
初回の較正は探索範囲の上限が4.0で**境界に張り付いていた**（walk-forward の5foldは全て4.5を
選んでおり、範囲が狭かったと分かった）。ここでは境界を外した範囲で測り直す。

  PYTHONIOENCODING=utf-8 python scripts/calibrate_mate_boost.py --emit
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

GRID = [3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 5.5]


def main() -> None:
    ap = argparse.ArgumentParser(description="mate_boost の較正")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--limit", type=int, default=0, help="直近N レースに絞る（0=全件）")
    ap.add_argument("--emit", action="store_true", help="branch_stats_men.json を更新")
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

    model, _elo, _lbl = load_for(False)
    raw = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(raw, args.db, men_features())
    if args.limit:
        samples = samples[-args.limit:]

    rows = []
    for s in samples:
        d, P = nb.get(s.race_id), pos.get(s.race_id)
        if not d or not P or 1 not in P or 2 not in P:
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
        lo = {x: i for i, m in enumerate(lines) for x in m}
        if lo.get(P[1]) is None or lo.get(P[2]) is None:
            continue
        rows.append((st, bs[0], lines, lo, int(lo[P[1]] == lo[P[2]])))

    act = sum(r[4] for r in rows) / len(rows)
    print(f"n={len(rows):,}  実測ライン決着 {act*100:.2f}%\n")
    best = None
    for mb in GRID:
        tot = 0.0
        for st, b, lines, lo, _ in rows:
            dd = branch_trifecta(st, b, lines, mate_boost=mb)
            tot += sum(p for (a, b2, _), p in dd.items()
                       if lo.get(a) is not None and lo.get(a) == lo.get(b2))
        pred = tot / len(rows)
        d = pred - act
        print(f"  mate_boost={mb:<5} 予測 {pred*100:5.2f}%  ズレ {d*100:+.2f}pt")
        if best is None or abs(d) < abs(best[1]):
            best = (mb, d)
    print(f"\n→ 最良 mate_boost={best[0]} （ズレ {best[1]*100:+.2f}pt）")
    if best[0] in (GRID[0], GRID[-1]):
        print("  ⚠ 範囲の端を引いた。GRIDを広げて再実行すること")
    if args.emit:
        st = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        old = st.get("mate_boost")
        st["mate_boost"] = best[0]
        st["mate_boost_note"] = (
            "1着になった車の番手に掛ける2着重みの倍率(1+m)。役割倍率だけでは「同じラインが"
            "1・2着を占める」同時性を作れず、真の主導権者を与えてもライン決着が17.8pt低いまま"
            f"だった。実測{len(rows):,}Rのライン決着{act*100:.1f}%に一致するよう較正して m={best[0]}"
            f"（ズレ{best[1]*100:+.2f}pt）。walk-forward 5foldも全て同値を選ぶ。手で決めた値ではない。")
        STATS_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"更新: {STATS_PATH.name}  mate_boost {old} → {best[0]}")


if __name__ == "__main__":
    main()
