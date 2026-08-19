"""レース種別(予選/準決勝/決勝/選抜/一般/特選=勝ち上がり構造)別の決着傾向。

番組表の「勝ち上がり条件」は race_name の種別＋競輪標準ルールで決まる(DBのrace_nameに格納済)。
準決勝=1着〜上位が決勝の高ステークス戦、決勝=上位選手、一般/選抜=最終日の敗者戦、等。
種別で走りの本気度・決着が変わるかを DBのみ(7車)で見る: 万車券率・勝者脚質・勝者ライン位置。

  PYTHONIOENCODING=utf-8 python scripts/analyze_race_role.py            # 男子
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

MAN = 10000


def _role(name):
    if not name:
        return "他"
    for kw, lab in [("準決", "準決勝"), ("決勝", "決勝"), ("予選", "予選"),
                    ("選抜", "選抜"), ("特選", "特選"), ("一般", "一般")]:
        if kw in name:
            return lab
    return "他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    role = {rid: _role(nm) for rid, nm, fs in
            c.execute("SELECT race_id,race_name,field_size FROM races") if fs == 7}
    leg = {}
    for rid, car, lt in c.execute("SELECT race_id,car_number,leg_type FROM entries"):
        leg[(rid, car)] = lt
    pay = {}
    for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        pay[rid] = (tuple(int(x) for x in combo.split("-")), p)
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

    def pos(car, ln):
        for m in ln or []:
            if car in m:
                return "単騎" if len(m) == 1 else ("先頭" if m.index(car) == 0 else ("番手" if m.index(car) == 1 else "後"))
        return "単騎"

    st = defaultdict(lambda: {"n": 0, "man": 0, "wleg": Counter(), "wpos": Counter()})
    for rid, (combo, p) in pay.items():
        r = role.get(rid)
        if r is None:
            continue
        s = st[r]
        s["n"] += 1
        if p >= MAN:
            s["man"] += 1
            s["wleg"][leg.get((rid, combo[0]))] += 1
            s["wpos"][pos(combo[0], lines.get(rid))] += 1

    print(f"レース種別(勝ち上がり)別の決着傾向  DB={Path(args.db).stem} 7車\n")
    print(f"   {'種別':<6}{'R数':>7}{'万車券率':>9}{'勝者逃':>8}{'勝者追':>8}{'勝者先頭':>9}{'勝者番手':>9}{'勝者単騎':>9}")
    for r in ["予選", "準決勝", "決勝", "選抜", "一般", "特選"]:
        s = st.get(r)
        if not s or s["n"] < 200:
            continue
        m = s["man"] or 1
        wl, wp = s["wleg"], s["wpos"]
        def pc(d, k): return f"{d.get(k,0)/m*100:.0f}%"
        print(f"   {r:<6}{s['n']:>7}{s['man']/s['n']*100:>8.1f}%{pc(wl,'逃'):>8}{pc(wl,'追'):>8}"
              f"{pc(wp,'先頭'):>9}{pc(wp,'番手'):>9}{pc(wp,'単騎'):>9}")


if __name__ == "__main__":
    main()
