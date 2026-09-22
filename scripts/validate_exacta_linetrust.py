"""二車単 ◎→番手(ライン信頼)の細分ドリル(男子7車)。

全体検証で「標準帯×◎→番手=93.1%」が最高セル。100%超の薄いスライスが有るか、
  ・◎勝率の細帯  ・番手のモデル勝率順位(2番手が強いライン=信頼)  ・◎ライン長(2/≥3)
で切って探す。himo不要(◎=argmax strength, 番手=ライン直後)なので高速。実払戻で決済。

  PYTHONIOENCODING=utf-8 python scripts/validate_exacta_linetrust.py --days 180
"""
from __future__ import annotations

import argparse
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
    mate, llen = {}, {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        ls = [x for x in ls if x]
        for ln in ls:
            for k in range(len(ln) - 1):
                mate[(rid, ln[k])] = ln[k + 1]
            for car in ln:
                llen[(rid, car)] = len(ln)
    return exa, mate, llen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()
    exa, mate, llen = _load(args.db)
    model, elo, lbl = load_for(False)
    base = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, model.feature_names)
    dated = [(s, _adate(s.race_id)) for s in samples if s.race_id in exa]
    days = sorted({d for _, d in dated})[-args.days:]
    target = [s for s, d in dated if d in days]
    print(f"男子7車 直近{args.days}日 {days[0]}〜{days[-1]}  ◎→番手ドリル {len(target)}レース\n")

    by_pw = defaultdict(lambda: {"pts": 0, "ret": 0, "hit": 0})
    by_rank = defaultdict(lambda: {"pts": 0, "ret": 0, "hit": 0})
    by_len = defaultdict(lambda: {"pts": 0, "ret": 0, "hit": 0})
    for s in target:
        rid = s.race_id
        st = model.strengths(s.X, s.car_numbers)
        anchor = max(st, key=st.get)
        nb = mate.get((rid, anchor))
        if not nb:
            continue
        tot = sum(st.values())
        pw = st[anchor] / tot
        # 番手のモデル勝率順位(1=最強)
        rank = sorted(st, key=st.get, reverse=True).index(nb) + 1
        ll = llen.get((rid, anchor), 2)
        combo, pay = exa[rid]
        hit = tuple(combo) == (anchor, nb)

        pwb = ("①<0.30" if pw < 0.30 else "②0.30-0.35" if pw < 0.35 else
               "③0.35-0.40" if pw < 0.40 else "④0.40-0.50" if pw < 0.50 else "⑤≥0.50")
        rb = "番手=2位" if rank == 2 else "番手=3位" if rank == 3 else "番手=4位以下"
        lb = f"ライン{ll}車" if ll <= 3 else "ライン4車+"
        for d in (by_pw[pwb], by_rank[rb], by_len[lb]):
            d["pts"] += 1
            if hit:
                d["hit"] += 1; d["ret"] += pay

    def show(title, dd, order):
        print(f"■ {title}")
        print(f"   {'区分':<14}{'点数':>7}{'的中率':>9}{'回収率':>9}")
        for k in order:
            d = dd.get(k)
            if not d or d["pts"] == 0:
                continue
            roi = d["ret"] / (d["pts"] * 100) * 100
            hr = d["hit"] / d["pts"] * 100
            flag = "  ★>100%" if roi >= 100 else ""
            print(f"   {k:<14}{d['pts']:>7}{hr:>8.1f}%{roi:>8.1f}%{flag}")
        print()

    show("◎勝率帯 別", by_pw, ["①<0.30", "②0.30-0.35", "③0.35-0.40", "④0.40-0.50", "⑤≥0.50"])
    show("番手のモデル順位 別", by_rank, ["番手=2位", "番手=3位", "番手=4位以下"])
    show("◎ライン長 別", by_len, ["ライン2車", "ライン3車", "ライン4車+"])


if __name__ == "__main__":
    main()
