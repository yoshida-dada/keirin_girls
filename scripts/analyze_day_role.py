"""男子: 開催日目 × 種別 で「狙う選手(勝者脚質)」が変わるかを確認。

先の知見: 後半日ほど荒れる/追込増(analyze_meet_day)、種別で決着激変(準決勝/決勝→先行, 選抜/一般→追込;
analyze_race_role)。日効果は種別経由(後半日=敗者戦増)なのか、同一種別内でも日で変わるのかを見る。
＝買い目の狙い目は「日別」か「種別別」か。

  PYTHONIOENCODING=utf-8 python scripts/analyze_day_role.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR


def _role(name):
    if not name:
        return "他"
    for kw in ["準決", "決勝", "予選", "選抜", "特選", "一般"]:
        if kw in name:
            return "準決勝" if kw == "準決" else kw
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
    win = {}
    for rid, combo in c.execute("SELECT race_id,combo FROM payouts_trifecta"):
        win[rid] = int(combo.split("-")[0])
    c.close()

    # (種別,日目) -> 勝者脚質分布
    cell = defaultdict(lambda: {"n": 0, "leg": Counter()})
    for rid, w in win.items():
        r = role.get(rid)
        if r is None:
            continue
        day = int(rid[10:12])
        if day > 3:
            continue
        s = cell[(r, day)]
        s["n"] += 1
        s["leg"][leg.get((rid, w))] += 1

    print("男子7車: 勝者脚質を 種別×日目 で（日効果は種別経由か・同一種別内で日差があるか）\n")
    print(f"   {'種別':<6}{'日目':>4}{'R数':>7}{'勝者逃':>8}{'勝者両':>8}{'勝者追':>8}")
    for r in ["予選", "準決勝", "決勝", "選抜", "一般", "特選"]:
        shown = False
        for day in (1, 2, 3):
            s = cell.get((r, day))
            if not s or s["n"] < 60:
                continue
            m = s["n"]
            def pc(k): return f"{s['leg'].get(k,0)/m*100:.0f}%"
            print(f"   {r:<6}{day:>4}{m:>7}{pc('逃'):>8}{pc('両'):>8}{pc('追'):>8}")
            shown = True
        if shown:
            print()


if __name__ == "__main__":
    main()
