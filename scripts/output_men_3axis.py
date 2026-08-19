"""本日(最新確定日)の男子全レースで、展開分岐買い目の before(現行) vs after(完全3軸調整) を比較。

before = build_branches の merged(展開分岐統合買い目)。本番と同一。
after  = クラス×種別×日目トレンドで軸調整:
  ・チャレンジ(A3) 準決勝/決勝 = 先行超堅い → 1着を逃/両に絞り点数圧縮
  ・敗者戦(選抜/一般/特選) = 追込狙い → 追込頭の目を補完（S級A級最終日はさらに厚く）
  ・S級 = 競争的 → 全体top補完で手広く
  ・その他(予選等) = 現行のまま
実払戻(payouts_trifecta)で決済し、点数差分・的中・回収を出す。

  PYTHONIOENCODING=utf-8 python scripts/output_men_3axis.py
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
from src.features.rider_narabi import compute_narabi_features
from src.features.venue_region import venue_name
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
    args = ap.parse_args()
    rn, cls, leg, tri, lines = _load(args.db)
    model, elo, lbl = load_for(False)
    bs = load_backstretch(is_girls=False)

    base = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    narabi = compute_narabi_features(args.db)

    dated = [(s, _adate(s.race_id)) for s in samples if s.race_id in tri and s.race_id in lines]
    if not dated:
        print("対象なし"); return
    today = max(d for _, d in dated)
    day = sorted([s for s, d in dated if d == today], key=lambda s: (s.race_id[:2], int(s.race_id[12:])))
    print(f"男子 本日(最新確定日) {today}  {len(day)}レース  before(現行) vs after(3軸調整)\n")

    tot = {"b_pts": 0, "b_ret": 0, "b_hit": 0, "a_pts": 0, "a_ret": 0, "a_hit": 0, "diff": 0}
    for s in day:
        rid = s.race_id
        st = model.strengths(s.X, s.car_numbers)
        pB = bs.strengths(s.X, s.car_numbers) if bs else None
        ln = lines[rid]
        before = _merged(build_branches(st, ln, pB))
        if not before:
            continue
        npos = {cc: narabi.get((rid, cc), {}).get("narabi_pos") for cc in s.car_numbers}
        dist = corrected_trifecta_probs(st, npos, MEN_PARAMS, lines=ln)
        ranked = [o for o, _ in sorted(dist.items(), key=lambda kv: -kv[1])]
        cl, role = _cls(cls.get(rid, [])), _role(rn.get(rid))

        def lg(car):
            return leg.get((rid, car))
        after = set(before)
        if cl == "チャレンジ" and role in ("準決勝", "決勝"):
            f = [o for o in before if lg(o[0]) in ("逃", "両")]
            base_set = set(f) if len(f) >= 4 else set(before)
            after = set(sorted(base_set, key=lambda o: -dist.get(o, 0))[:max(4, math.ceil(len(base_set) * 0.7))])
            adj = "先行軸で絞る"
        elif role in ("選抜", "一般", "特選"):
            tsuiko = [o for o in ranked if lg(o[0]) == "追"][:6]
            after = set(before) | set(tsuiko)
            adj = "追込頭を補完"
        elif cl == "S級":
            after = set(before) | set(ranked[:len(before) + 4])
            adj = "手広く"
        else:
            adj = "現行のまま"

        combo, pay = tri[rid]
        b_hit = tuple(combo) in before
        a_hit = tuple(combo) in after
        d = len(after) - len(before)
        tot["b_pts"] += len(before); tot["b_ret"] += pay if b_hit else 0; tot["b_hit"] += b_hit
        tot["a_pts"] += len(after); tot["a_ret"] += pay if a_hit else 0; tot["a_hit"] += a_hit
        tot["diff"] += abs(d)
        chg = "同" if d == 0 else (f"+{d}" if d > 0 else str(d))
        mark = "" if b_hit == a_hit else ("★after的中化" if a_hit else "▼after外れ化")
        print(f"{venue_name(rid[:2]) or rid[:2]:<5}R{int(rid[12:]):<2} [{cl}/{role}] {adj:<10} "
              f"点{len(before)}→{len(after)}({chg}) 結果{'-'.join(map(str,combo))} "
              f"before{'○' if b_hit else '×'}/after{'○' if a_hit else '×'} {mark}")

    n = len(day)
    def roi(p, r): return r / (p * 100) * 100 if p else 0
    print("\n" + "=" * 54)
    print(f"【集計】{n}レース  平均点数差分 {tot['diff']/n:+.1f}点")
    print(f"  before(現行) 点数計{tot['b_pts']:>4} 的中{tot['b_hit']}/{n} 払戻{tot['b_ret']:>7,}円 回収率{roi(tot['b_pts'],tot['b_ret']):.1f}%")
    print(f"  after(3軸調整) 点数計{tot['a_pts']:>4} 的中{tot['a_hit']}/{n} 払戻{tot['a_ret']:>7,}円 回収率{roi(tot['a_pts'],tot['a_ret']):.1f}%")


if __name__ == "__main__":
    main()
