"""展開分岐買い目「絞る-only」3軸調整の1ヶ月検証(男子)。

before = build_branches の merged(現行)。
after  = チャレンジ(A3)の準決勝/決勝のみ「先行軸(逃/両)に絞り点数圧縮」。他の全レースは現行のまま
        （敗者戦/S級の"広げる"は今回の検証で逆効果と判明したため一切しない＝引き算だけ）。
本番モデル+本番展開AIで予測、実払戻で決済。全体/チャレンジ高ステークス/その他 で before vs after を比較。

  PYTHONIOENCODING=utf-8 python scripts/validate_branch_shrink.py --days 30
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict, Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.model.backstretch import load_backstretch
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.development_branches import build_branches


def _adate(rid):
    return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=int(rid[10:12]) - 1)


def _role(name):
    if not name:
        return "他"
    for kw in ["準決", "決勝", "予選", "選抜", "特選", "一般"]:
        if kw in name:
            return "準決勝" if kw == "準決" else kw
    return "他"


def _cls(crs):
    m = Counter(crs).most_common(1)[0][0] if crs else None
    if m in ("SS", "S1", "S2"):
        return "S級"
    if m in ("A1", "A2"):
        return "A級"
    if m == "A3":
        return "チャレンジ"
    return "他"


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    rn, cls, leg = {}, defaultdict(list), {}
    for rid, nm in c.execute("SELECT race_id,race_name FROM races"):
        rn[rid] = nm
    for rid, car, lt, cr in c.execute("SELECT race_id,car_number,leg_type,class_rank FROM entries"):
        leg[(rid, car)] = lt
        cls[rid].append(cr)
    tri = {rid: (tuple(int(x) for x in combo.split("-")), p)
           for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    c.close()
    lines = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        lines[rid] = [x for x in ls if x]
    return rn, cls, leg, tri, lines


def _merged(br):
    cs = set()
    for f in (br or {}).get("merged", {}).get("forms", []):
        cs |= {(a, b, c) for a in f["first"] for b in f["second"] for c in f["third"]
               if len({a, b, c}) == 3}
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    rn, cls, leg, tri, lines = _load(args.db)
    model, elo, lbl = load_for(False)
    bs = load_backstretch(is_girls=False)

    base = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)

    dated = [(s, _adate(s.race_id)) for s in samples if s.race_id in tri and s.race_id in lines]
    days = sorted({d for _, d in dated})[-args.days:]
    target = [s for s, d in dated if d in days]
    print(f"男子 直近{args.days}日 {days[0]}〜{days[-1]}  対象{len(target)}レース\n")

    agg = defaultdict(lambda: {"b_pts": 0, "b_ret": 0, "b_hit": 0, "a_pts": 0, "a_ret": 0, "a_hit": 0, "n": 0})
    for s in target:
        rid = s.race_id
        st = model.strengths(s.X, s.car_numbers)
        pB = bs.strengths(s.X, s.car_numbers) if bs else None
        ln = lines[rid]
        before = _merged(build_branches(st, ln, pB))
        if not before:
            continue
        cl, role = _cls(cls.get(rid, [])), _role(rn.get(rid))
        # after: チャレンジ準決勝/決勝のみ先行軸で絞る、他は現行のまま
        is_shrink = (cl == "チャレンジ" and role in ("準決勝", "決勝"))
        if is_shrink:
            dist = corrected_trifecta_probs(st, {}, MEN_PARAMS, lines=ln)
            f = [o for o in before if leg.get((rid, o[0])) in ("逃", "両")]
            base_set = set(f) if len(f) >= 4 else set(before)
            after = set(sorted(base_set, key=lambda o: -dist.get(o, 0))[:max(4, math.ceil(len(base_set) * 0.7))])
        else:
            after = before
        combo, pay = tri[rid]
        b_hit = tuple(combo) in before
        a_hit = tuple(combo) in after
        grp = "チャレンジ準決勝決勝" if is_shrink else "その他(不変)"
        for key in (grp, "全体"):
            a = agg[key]
            a["n"] += 1
            a["b_pts"] += len(before); a["b_ret"] += pay if b_hit else 0; a["b_hit"] += b_hit
            a["a_pts"] += len(after); a["a_ret"] += pay if a_hit else 0; a["a_hit"] += a_hit

    def roi(p, r): return r / (p * 100) * 100 if p else 0
    print(f"{'区分':<20}{'R数':>5}{'現行点数':>8}{'現行的中':>8}{'現行ROI':>9}"
          f"{'絞点数':>8}{'絞的中':>8}{'絞ROI':>9}{'ROI差':>8}")
    for key in ["全体", "チャレンジ準決勝決勝", "その他(不変)"]:
        a = agg.get(key)
        if not a or a["n"] == 0:
            continue
        n = a["n"]
        br, ar = roi(a["b_pts"], a["b_ret"]), roi(a["a_pts"], a["a_ret"])
        print(f"{key:<20}{n:>5}{a['b_pts']/n:>7.1f}{a['b_hit']/n*100:>7.1f}%{br:>8.1f}%"
              f"{a['a_pts']/n:>7.1f}{a['a_hit']/n*100:>7.1f}%{ar:>8.1f}%{ar-br:>+7.1f}")


if __name__ == "__main__":
    main()
