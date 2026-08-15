"""mate_boost / line_boost（1着と同ラインの選手に掛ける2着重み）を実測に合わせて較正する。

**2つの係数を2つの目標に同時に合わせる。** 合計（ライン決着率）だけに合わせると配分を誤る:
mate_boost 単独で較正したとき、合計は合う(55.6% vs 実測55.3%)のに
番手 42.2%(実測32.2%) / 同ライン他 13.4%(実測23.1%) と ±10pt ずれていた。
目標は「勝者の番手が2着」と「勝者と同ライン他が2着」の2つの実測シェア。

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

MB_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LB_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


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
        rows.append((st, bs[0], lines, lo, int(lo[P[1]] == lo[P[2]]), P[1], P[2]))

    # 目標は2つ: 勝者の番手が2着 / 勝者と同ライン他が2着 の実測シェア
    def cond_role(x, w, lines):
        lo = {c2: i for i, m in enumerate(lines) for c2 in m}
        lw, lx = lo.get(w), lo.get(x)
        if lx is None or lw is None or lx != lw:
            return "other"
        mem = lines[lw]
        i = mem.index(w)
        return "mate" if (i + 1 < len(mem) and mem[i + 1] == x) else "same"

    a_mate = a_same = 0
    for st, b, lines, lo, _s, w, s2 in rows:
        r = cond_role(s2, w, lines)
        a_mate += int(r == "mate")
        a_same += int(r == "same")
    n = len(rows)
    t_mate, t_same = a_mate / n, a_same / n
    print(f"n={n:,}  実測: 勝者の番手 {t_mate*100:.2f}% / 同ライン他 {t_same*100:.2f}% "
          f"（合計 {(t_mate+t_same)*100:.2f}%）\n")

    best = None
    print(f"{'mate':>6}{'line':>6}{'番手':>9}{'同ライン他':>11}{'誤差和':>9}")
    for mb in MB_GRID:
        for lb in LB_GRID:
            pm = ps = 0.0
            for st, b, lines, lo, _s, w, _s2 in rows:
                dd = branch_trifecta(st, b, lines, mate_boost=mb, line_boost=lb)
                sub = {}
                for (x1, x2, _x3), p in dd.items():
                    if x1 == w:
                        sub[x2] = sub.get(x2, 0.0) + p
                z = sum(sub.values())
                if z <= 0:
                    continue
                for x, p in sub.items():
                    r = cond_role(x, w, lines)
                    if r == "mate":
                        pm += p / z
                    elif r == "same":
                        ps += p / z
            pm /= n
            ps /= n
            err = abs(pm - t_mate) + abs(ps - t_same)
            print(f"{mb:>6.1f}{lb:>6.1f}{pm*100:>8.1f}%{ps*100:>10.1f}%{err*100:>8.2f}")
            if best is None or err < best[2]:
                best = (mb, lb, err, pm, ps)
    print(f"\n→ 最良 mate_boost={best[0]} / line_boost={best[1]} "
          f"（番手 {best[3]*100:.1f}% / 同ライン他 {best[4]*100:.1f}% / 誤差和 {best[2]*100:.2f}pt）")
    if best[0] in (MB_GRID[0], MB_GRID[-1]) or best[1] in (LB_GRID[0], LB_GRID[-1]):
        print("  ⚠ 範囲の端を引いた。GRIDを広げて再実行すること")
    if args.emit:
        st = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        st["mate_boost"] = best[0]
        st["line_boost"] = best[1]
        st["boost_note"] = (
            "1着と同ラインの選手に掛ける2着重み。番手(mate)とそれ以外(line)を別係数にする。"
            "合計だけに合わせると配分を誤る（mate単独較正時は合計55.6%で合うのに"
            "番手42.2%/同ライン他13.4%と実測32.2%/23.1%から±10ptずれた）。"
            f"実測{n:,}Rの2つのシェアへ同時に較正して mate={best[0]} / line={best[1]}"
            f"（番手{best[3]*100:.1f}% / 同ライン他{best[4]*100:.1f}%）。手で決めた値ではない。")
        STATS_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"更新: {STATS_PATH.name}")


if __name__ == "__main__":
    main()
