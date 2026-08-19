"""会場・最新開催日で「展開分岐の統合買い目(現行) vs 種別調整＋三連複2点(新)」を比較(男子)。

現行 = predict_race と同じ build_branches の merged(展開分岐をまたぐ統合買い目, フォーメーション和集合)。
新   = その merged を種別(勝ち上がり)で調整—準決勝/決勝=補正確率で上位70%に絞る / 選抜・一般=全体top補完で
       広げる / 予選・特選=そのまま—＋三連複2点(保険)。実払戻で決済。

  PYTHONIOENCODING=utf-8 python scripts/output_venue_branches.py --venue-code 45
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
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.model.backstretch import load_backstretch
from src.features.rider_narabi import compute_narabi_features
from src.features.venue_region import venue_name
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.development_branches import build_branches


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
    nm = {}
    for rid, car, name in c.execute("SELECT race_id,car_number,rider_name FROM entries"):
        nm[(rid, car)] = name
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
    return rn, nm, tri, trio, lines


def _merged_combos(merged):
    cs = set()
    for f in (merged or {}).get("forms", []):
        cs |= {(a, b, c) for a in f["first"] for b in f["second"] for c in f["third"]
               if len({a, b, c}) == 3}
    return cs


def _cs(t):
    return "-".join(str(x) for x in t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue-code", default="45")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    args = ap.parse_args()
    rn, nmap, tri, trio, lines = _load(args.db)
    model, elo, lbl = load_for(False)
    bs = load_backstretch(is_girls=False)

    base = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    narabi = compute_narabi_features(args.db)

    vc = args.venue_code
    vs = [s for s in samples if s.race_id[:2] == vc and s.race_id in tri and s.race_id in lines]
    if not vs:
        print("該当レースなし（ライン/払戻あり）"); return
    latest = max(s.race_id[2:12] for s in vs)
    day = sorted([s for s in vs if s.race_id[2:12] == latest], key=lambda s: int(s.race_id[12:]))
    print(f"{venue_name(vc) or vc}競輪 開催初日{latest[:8]} {int(latest[8:10])}日目  {len(day)}レース"
          f"（展開分岐買い目・現行 vs 新=種別調整＋三連複2）\n")

    tot = {"c_pts": 0, "c_ret": 0, "c_hit": 0, "n_pts": 0, "n_ret": 0, "n_hit": 0}
    for s in day:
        st = model.strengths(s.X, s.car_numbers)
        pB = bs.strengths(s.X, s.car_numbers) if bs else None
        ln = lines[s.race_id]
        npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
        dist = corrected_trifecta_probs(st, npos, MEN_PARAMS, lines=ln)
        br = build_branches(st, ln, pB, names={c: nmap.get((s.race_id, c)) for c in s.car_numbers})
        if not br:
            print(f"── R{int(s.race_id[12:])}: 展開分岐生成不可（スキップ）"); continue
        cur = _merged_combos(br.get("merged"))
        if not cur:
            continue
        combo, pay = tri[s.race_id]
        r = _role(rn.get(s.race_id))
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        # 新: 種別調整
        if r == "堅":
            keep = sorted(cur, key=lambda o: -dist.get(o, 0))[:max(4, math.ceil(len(cur) * 0.7))]
            ntri = set(keep)
        elif r == "荒":
            ntri = set(cur)
            for o, _ in ranked:
                if len(ntri) >= len(cur) + 4:
                    break
                ntri.add(o)
        else:
            ntri = set(cur)
        trp = defaultdict(float)
        for (a, b, c), p in dist.items():
            trp[frozenset((a, b, c))] += p
        ntrio = [o for o, _ in sorted(trp.items(), key=lambda kv: -kv[1])[:2]]

        c_hit = tuple(combo) in cur
        n_tri_hit = tuple(combo) in ntri
        n_trio_hit = (s.race_id in trio) and (trio[s.race_id][0] in ntrio)
        n_pts = len(ntri) + len(ntrio)
        n_ret = (pay if n_tri_hit else 0) + (trio[s.race_id][1] if n_trio_hit else 0)
        tot["c_pts"] += len(cur); tot["c_ret"] += pay if c_hit else 0; tot["c_hit"] += c_hit
        tot["n_pts"] += n_pts; tot["n_ret"] += n_ret; tot["n_hit"] += int(n_tri_hit or n_trio_hit)

        rlab = {"堅": "準決勝/決勝(絞る)", "荒": "選抜/一般(広げる)", "標": "予選/特選"}[r]
        print(f"── R{int(s.race_id[12:])} {rn.get(s.race_id,'')} [{rlab}] 結果{_cs(combo)}({pay:,}円)")
        print(f"   現行(展開分岐{len(cur)}点): {'的中 '+format(pay,',')+'円' if c_hit else '不的中'}")
        print(f"   新(三連単{len(ntri)}点+複2): 三連単{'的中'if n_tri_hit else '×'}"
              + (f"／三連複{'的中'if n_trio_hit else '×'}" if s.race_id in trio else "／複データ無")
              + f"  → {n_ret:,}円" if n_ret else f"  → 不的中")
        print()

    n = len(day)
    def roi(p, r): return r / (p * 100) * 100 if p else 0
    print("=" * 52)
    print("【集計】展開分岐買い目")
    print(f"  現行(統合買い目)  点数計{tot['c_pts']:>4} 的中{tot['c_hit']}/{n} "
          f"払戻{tot['c_ret']:>7,}円 回収率{roi(tot['c_pts'],tot['c_ret']):.1f}%")
    print(f"  新(種別調整＋複2) 点数計{tot['n_pts']:>4} 的中{tot['n_hit']}/{n} "
          f"払戻{tot['n_ret']:>7,}円 回収率{roi(tot['n_pts'],tot['n_ret']):.1f}%")


if __name__ == "__main__":
    main()
