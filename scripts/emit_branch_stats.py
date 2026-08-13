"""(c) 展開分岐の材料を実測して JSON に焼き出す。

主導権(B)を取る選手が決まると、着順の分布は「B本人 / B番手 / 他ライン先頭 / …」という
**役割**でよく説明できる（実測 25,238R）。そこで分岐ごとの着順分布を
  s'_c = s_c × w[役割(c | B=b)]
という役割倍率で作る。倍率は「実測の役割別シェア ÷ 素のモデルが出す役割別シェア」で、
**モデルの実力評価はそのまま残しつつ、役割による偏りだけを実測に合わせる**。

倍率を手で決めない（それをやると LEG_AGGR の二の舞になる）。1着・2着・3着で別々に測る。

  PYTHONIOENCODING=utf-8 python scripts/emit_branch_stats.py --emit src/model/branch_stats_men.json
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
from src.model.plackett_luce import all_trifecta_probs

ROLES = ["B本人", "B番手", "B同ライン他", "他ライン先頭", "他ライン番手", "他ライン3番手+", "単騎"]


def role_of(car: int, b: int, lines: dict[int, list[int]], line_of: dict[int, int]) -> str:
    """主導権者 b から見た car の役割。"""
    lb = line_of.get(b)
    lc = line_of.get(car)
    if lc is None or lb is None:
        return "単騎"
    if lc == lb:
        if car == b:
            return "B本人"
        mem = lines[lb]
        return "B番手" if mem.index(car) == mem.index(b) + 1 else "B同ライン他"
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
        pos[rid][car] = p
        sbm[rid][car] = s
    c.close()
    return nb, pos, sbm


def main() -> None:
    ap = argparse.ArgumentParser(description="展開分岐の役割倍率を実測")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--emit")
    args = ap.parse_args()

    nb, pos, sbm = _ctx(args.db)
    model, _elo, lbl = load_for(False)
    if model is None:
        raise SystemExit("男子モデルが無い")
    raw = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(raw, args.db, men_features())
    print(f"{lbl}モデルで {len(samples):,}レースを走査")

    act = [defaultdict(int) for _ in range(3)]     # 実測: 着順ごとの役割カウント
    mdl = [defaultdict(float) for _ in range(3)]   # モデル: 同じ役割の確率和
    n = 0
    for s in samples:
        rid = s.race_id
        d = nb.get(rid)
        P = pos.get(rid)
        if not d or not P:
            continue
        bs = [c2 for c2 in d if sbm[rid].get(c2) and "B" in str(sbm[rid][c2])]
        if len(bs) != 1:
            continue
        b = bs[0]
        mem = defaultdict(list)
        for car, (li, pi) in d.items():
            mem[li].append((pi, car))
        lines = {li: [c2 for _, c2 in sorted(v)] for li, v in mem.items()}
        line_of = {c2: li for li, v in lines.items() for c2 in v}
        st = model.strengths(s.X, s.car_numbers)
        if not st:
            continue
        n += 1
        # 実測
        for k in range(3):
            car = next((c2 for c2, p in P.items() if p == k + 1), None)
            if car is not None and car in d:
                act[k][role_of(car, b, lines, line_of)] += 1
        # モデル（素のPL）の役割別シェア。2着/3着は逐次確率を積み上げる
        tri = all_trifecta_probs(st)
        for (a1, a2, a3), p in tri.items():
            for k, car in enumerate((a1, a2, a3)):
                if car in d:
                    mdl[k][role_of(car, b, lines, line_of)] += p

    out = {"generated": "auto", "n_races": n, "roles": ROLES, "weights": {}}
    print(f"\n{'着':>3}{'役割':>14}{'実測':>9}{'モデル':>9}{'倍率':>8}")
    for k in range(3):
        ta = sum(act[k].values()) or 1
        tm = sum(mdl[k].values()) or 1.0
        w = {}
        for r in ROLES:
            a = act[k].get(r, 0) / ta
            m = mdl[k].get(r, 0.0) / tm
            w[r] = round(a / m, 4) if m > 1e-9 else 1.0
            print(f"{k+1:>3}{r:>14}{a*100:>8.1f}%{m*100:>8.1f}%{w[r]:>8.2f}")
        out["weights"][str(k + 1)] = w
        print()
    print(f"n = {n:,}レース")
    if args.emit:
        Path(args.emit).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"書き出し: {args.emit}")


if __name__ == "__main__":
    main()
