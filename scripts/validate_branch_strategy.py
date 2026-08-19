"""展開分岐買い目の 現行(統合買い目) vs 新(種別調整＋三連複2) を直近N日で検証(男子)。

現行 = build_branches の merged(展開分岐をまたぐ統合買い目=フォーメーション和集合)。本番と同一。
新   = merged を種別で調整(準決勝/決勝=補正上位70%に絞る/選抜・一般=全体top補完で広げる/予選特選=同)＋三連複2点。
本番モデル(load_for)＋本番展開AIで予測、実払戻(payouts_trifecta/_trio)で決済。両者同じ予測で買い方だけ変える。

  PYTHONIOENCODING=utf-8 python scripts/validate_branch_strategy.py --days 30
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.model.backstretch import load_backstretch
from src.features.rider_narabi import compute_narabi_features
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.development_branches import build_branches


def _actual_date(rid):
    return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=int(rid[10:12]) - 1)


def _role(name):
    if not name:
        return "標"
    for kw, lab in [("準決", "堅"), ("決勝", "堅"), ("予選", "標"),
                    ("選抜", "荒"), ("一般", "荒"), ("特選", "標")]:
        if kw in name:
            return lab
    return "標"


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    rn = {rid: nm for rid, nm in c.execute("SELECT race_id,race_name FROM races")}
    tri, trio = {}, {}
    for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        tri[rid] = (tuple(int(x) for x in combo.split("-")), p)
    for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trio"):
        trio[rid] = (frozenset(int(x) for x in combo.replace("=", "-").split("-")), p)
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
    return rn, tri, trio, lines


def _merged_combos(merged):
    cs = set()
    for f in (merged or {}).get("forms", []):
        cs |= {(a, b, c) for a in f["first"] for b in f["second"] for c in f["third"]
               if len({a, b, c}) == 3}
    return cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    rn, tri, trio, lines = _load(args.db)
    model, elo, lbl = load_for(False)
    bs = load_backstretch(is_girls=False)

    base = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    narabi = compute_narabi_features(args.db)

    dated = [(s, _actual_date(s.race_id)) for s in samples
             if s.race_id in tri and s.race_id in lines]
    days = sorted({d for _, d in dated})[-args.days:]
    target = [s for s, d in dated if d in days]
    print(f"男子 直近{args.days}日 {days[0]}〜{days[-1]}  対象{len(target)}レース\n")

    agg = defaultdict(lambda: {"c_pts": 0, "c_ret": 0, "c_hit": 0, "n_pts": 0, "n_ret": 0, "n_hit": 0, "n": 0})
    ntrio_settleable = 0
    for s in target:
        st = model.strengths(s.X, s.car_numbers)
        pB = bs.strengths(s.X, s.car_numbers) if bs else None
        ln = lines[s.race_id]
        br = build_branches(st, ln, pB)
        cur = _merged_combos(br.get("merged")) if br else set()
        if not cur:
            continue
        npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
        dist = corrected_trifecta_probs(st, npos, MEN_PARAMS, lines=ln)
        combo, pay = tri[s.race_id]
        r = _role(rn.get(s.race_id))
        ranked = [o for o, _ in sorted(dist.items(), key=lambda kv: -kv[1])]
        if r == "堅":
            ntri = set(sorted(cur, key=lambda o: -dist.get(o, 0))[:max(4, math.ceil(len(cur) * 0.7))])
        elif r == "荒":
            ntri = set(cur)
            for o in ranked:
                if len(ntri) >= len(cur) + 4:
                    break
                ntri.add(o)
        else:
            ntri = set(cur)
        trp = defaultdict(float)
        for (a, b, c), p in dist.items():
            trp[frozenset((a, b, c))] += p
        ntrio = [o for o, _ in sorted(trp.items(), key=lambda kv: -kv[1])[:2]]

        for key in (r, "全"):
            a = agg[key]
            a["n"] += 1
            a["c_pts"] += len(cur); a["c_ret"] += pay if tuple(combo) in cur else 0
            a["c_hit"] += int(tuple(combo) in cur)
            ntri_hit = tuple(combo) in ntri
            has_trio = s.race_id in trio
            ntrio_hit = has_trio and (trio[s.race_id][0] in ntrio)
            a["n_pts"] += len(ntri) + len(ntrio)
            a["n_ret"] += (pay if ntri_hit else 0) + (trio[s.race_id][1] if ntrio_hit else 0)
            a["n_hit"] += int(ntri_hit or ntrio_hit)
        if s.race_id in trio:
            ntrio_settleable += 1

    def roi(p, r): return r / (p * 100) * 100 if p else 0
    print(f"（三連複払戻あり {ntrio_settleable}/{agg['全']['n']}レース＝新の三連複はこの分だけ決済）\n")
    print(f"{'区分':<16}{'R数':>5}{'現行点数':>8}{'現行的中':>8}{'現行ROI':>9}"
          f"{'新点数':>8}{'新的中':>8}{'新ROI':>9}")
    for key, lab in [("全", "全体"), ("堅", "準決勝/決勝(絞る)"), ("荒", "選抜/一般(広げる)"), ("標", "予選/特選")]:
        a = agg.get(key)
        if not a or a["n"] == 0:
            continue
        n = a["n"]
        print(f"{lab:<16}{n:>5}{a['c_pts']/n:>7.1f}{a['c_hit']/n*100:>7.1f}%{roi(a['c_pts'],a['c_ret']):>8.1f}%"
              f"{a['n_pts']/n:>7.1f}{a['n_hit']/n*100:>7.1f}%{roi(a['n_pts'],a['n_ret']):>8.1f}%")


if __name__ == "__main__":
    main()
