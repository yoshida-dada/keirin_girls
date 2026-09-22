"""二車単(exacta)の狙い目マップ＝ROI検証(男子)。プロジェクトの核心S5。

三連単では控除率25%+人気薄バイアスでROI>100%ゾーンが無かった。二車単は組合せが42(7車)と
少なく1点単価が高い(中央値×10.1)ので、控除の食い込みと本命-穴バイアスの効き方が違う可能性を検証。

本番モデル→himo補正の三連単確率を二車単に周辺化。戦略:
  top1/top3/top5 = モデル確率上位N点
  ◎→番手     = ◎(勝率最大)からライン番手(直後)への1点(ライン信頼)
◎勝率帯(軸堅/標準/混戦)でバケット化し、実払戻(payouts_exacta)で回収率を出す。

  PYTHONIOENCODING=utf-8 python scripts/validate_exacta_roi.py --days 90
"""
from __future__ import annotations

import argparse
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
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS


def _adate(rid):
    return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=int(rid[10:12]) - 1)


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    exa = {rid: (tuple(int(x) for x in combo.split("-")), p)
           for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_exacta")}
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    c.close()
    lines, mate = {}, {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        lines[rid] = [x for x in ls if x]
        for ln in lines[rid]:
            for k in range(len(ln) - 1):
                mate[(rid, ln[k])] = ln[k + 1]   # 直後(番手)
    return exa, lines, mate


def _exacta_probs(st, ln):
    """himo補正三連単→二車単(a,b)へ周辺化。"""
    tri = corrected_trifecta_probs(st, {}, MEN_PARAMS, lines=ln)
    ex = defaultdict(float)
    for (a, b, c), p in tri.items():
        ex[(a, b)] += p
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--days", type=int, default=90)
    args = ap.parse_args()
    exa, lines, mate = _load(args.db)
    model, elo, lbl = load_for(False)

    base = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    dated = [(s, _adate(s.race_id)) for s in samples if s.race_id in exa and s.race_id in lines]
    days = sorted({d for _, d in dated})[-args.days:]
    target = [s for s, d in dated if d in days]
    print(f"男子7車 直近{args.days}日 {days[0]}〜{days[-1]}  対象{len(target)}レース\n")

    strat = ["top1", "top3", "top5", "◎→番手"]
    # 全体集計 と ◎勝率帯別
    agg = defaultdict(lambda: {s: {"pts": 0, "ret": 0, "hit": 0} for s in strat} | {"n": 0})

    for s in target:
        rid = s.race_id
        st = model.strengths(s.X, s.car_numbers)
        ln = lines[rid]
        ex = _exacta_probs(st, ln)
        ranked = [k for k, _ in sorted(ex.items(), key=lambda kv: -kv[1])]
        combo, pay = exa[rid]
        combo = tuple(combo)
        anchor = max(st, key=st.get)
        pwin = st[anchor] / sum(st.values())
        band = "軸堅(◎≥40%)" if pwin >= 0.40 else ("混戦(◎<25%)" if pwin < 0.25 else "標準")

        picks = {"top1": ranked[:1], "top3": ranked[:3], "top5": ranked[:5]}
        nb = mate.get((rid, anchor))
        picks["◎→番手"] = [(anchor, nb)] if nb else []

        for key in (band, "全体"):
            a = agg[key]
            a["n"] += 1
            for sname, ps in picks.items():
                a[sname]["pts"] += len(ps)
                if combo in ps:
                    a[sname]["hit"] += 1
                    a[sname]["ret"] += pay

    def roi(d): return d["ret"] / (d["pts"] * 100) * 100 if d["pts"] else 0
    def hr(d, n): return d["hit"] / n * 100 if n else 0
    for key in ["全体", "軸堅(◎≥40%)", "標準", "混戦(◎<25%)"]:
        a = agg.get(key)
        if not a or a["n"] == 0:
            continue
        n = a["n"]
        print(f"■ {key}  {n}レース")
        print(f"   {'戦略':<8}{'点/R':>7}{'的中率':>9}{'回収率':>9}")
        for sname in strat:
            d = a[sname]
            if d["pts"] == 0:
                continue
            print(f"   {sname:<8}{d['pts']/n:>6.1f}{hr(d,n):>8.1f}%{roi(d):>8.1f}%")
        print()


if __name__ == "__main__":
    main()
