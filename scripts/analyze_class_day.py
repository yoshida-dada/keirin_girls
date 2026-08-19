"""男子: 日目 × 種別 × クラス の完全3軸トレンド（展開分岐買い目の狙う選手を精緻化）。

種別は概ね日固定(予選=初日/準決勝=中日/決勝=最終日)だが、敗者戦(選抜/一般/特選)は日をまたぐ。
そこでクラス×種別に「日目」を重ね、同一クラス・同一種別内で日により狙う脚質が変わるかを見る。

  PYTHONIOENCODING=utf-8 python scripts/analyze_class_day.py
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


def _cls(crs):
    m = Counter(crs).most_common(1)[0][0] if crs else None
    if m in ("SS", "S1", "S2"):
        return "S級"
    if m in ("A1", "A2"):
        return "A級"
    if m == "A3":
        return "チャレンジ"
    return "他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    args = ap.parse_args()
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    role = {rid: _role(nm) for rid, nm, fs in
            c.execute("SELECT race_id,race_name,field_size FROM races") if fs == 7}
    leg, cls = {}, defaultdict(list)
    for rid, car, lt, cr in c.execute("SELECT race_id,car_number,leg_type,class_rank FROM entries"):
        leg[(rid, car)] = lt
        cls[rid].append(cr)
    win = {rid: int(combo.split("-")[0]) for rid, combo in
           c.execute("SELECT race_id,combo FROM payouts_trifecta")}
    c.close()

    cell = defaultdict(lambda: {"n": 0, "leg": Counter()})
    for rid, r in role.items():
        if rid not in win:
            continue
        day = int(rid[10:12])
        if day > 3:
            continue
        key = (_cls(cls.get(rid, [])), r, day)
        s = cell[key]
        s["n"] += 1
        s["leg"][leg.get((rid, win[rid]))] += 1

    print("男子7車: クラス×種別×日目 の勝者脚質（同一クラス・種別で日差があるか＝敗者戦中心）\n")
    print(f"   {'クラス':<7}{'種別':<6}{'日目':>4}{'R数':>7}{'逃':>7}{'両':>7}{'追':>7}")
    for cl in ["S級", "A級", "チャレンジ"]:
        for r in ["予選", "準決勝", "決勝", "選抜", "一般", "特選"]:
            days = [d for d in (1, 2, 3) if cell.get((cl, r, d), {}).get("n", 0) >= 60]
            if len(days) < 2:      # 日差を見るのが目的なので複数日あるセルのみ
                continue
            for d in days:
                s = cell[(cl, r, d)]
                m = s["n"]
                def pc(k): return f"{s['leg'].get(k,0)/m*100:.0f}%"
                print(f"   {cl:<7}{r:<6}{d:>4}{m:>7}{pc('逃'):>7}{pc('両'):>7}{pc('追'):>7}")
            print()


if __name__ == "__main__":
    main()
