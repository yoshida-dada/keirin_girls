"""落車明けの成績低下を予備調査（男子7車）。

現状DBは落車を特定できず、落車/失格/欠車が全て position NULL(2736件)。ここでは「前走がDNF
(position NULL)」を落車の代理として、DNF後 数レースの成績残差(通算平均着−当該着,+=良化)が
下がるか、何レース(≒何開催)尾を引くかを見る。失格/欠車の混入で信号は薄まる向きなので、
それでも低下が出れば落車固有ではより強いはず。

  PYTHONIOENCODING=utf-8 python scripts/analyze_crash.py
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

DB = str(DATA_DIR / "keirin_men.sqlite")


def main():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    fs = {rid: n for rid, n in c.execute("SELECT race_id,field_size FROM races")}
    rdt = {rid: d for rid, d in c.execute("SELECT race_id,race_date FROM races")}
    rows = c.execute("SELECT race_id,rider_name,position FROM results"
                     " WHERE rider_name IS NOT NULL").fetchall()
    c.close()

    def pd(s):
        try:
            return date.fromisoformat(str(s))
        except (ValueError, TypeError):
            return None

    byrider = defaultdict(list)   # name -> [(date, rid, pos_or_None)]
    for rid, nm, pos in rows:
        if fs.get(rid) != 7:
            continue
        d = pd(rdt.get(rid))
        if d:
            byrider[nm].append((d, rid, pos))
    # 通算平均着（完走のみ）
    cbase = {}
    for nm, v in byrider.items():
        fin = [p for _, _, p in v if p is not None]
        if fin:
            cbase[nm] = sum(fin) / len(fin)

    # 各完走レースについて「直近のDNFから何レース後か」を数える
    agg = defaultdict(lambda: {"n": 0, "res": 0.0, "top3": 0, "win": 0})
    dnf_total = 0
    for nm, v in byrider.items():
        if nm not in cbase:
            continue
        v.sort()
        base = cbase[nm]
        since = None       # 直近DNFから何レース経ったか（Noneは直近にDNF無し）
        for d, rid, pos in v:
            if pos is None:                 # このレースがDNF
                since = 0
                dnf_total += 1
                continue
            # 完走レース: since に応じてバケット
            if since is None:
                key = "DNF無し(基準)"
            else:
                since += 1
                key = ("①明け1走目" if since == 1 else "②2走目" if since == 2 else
                       "③3走目" if since == 3 else "④4-6走目" if since <= 6 else "⑤7走目以降")
            a = agg[key]
            a["n"] += 1; a["res"] += base - pos; a["top3"] += int(pos <= 3); a["win"] += int(pos == 1)

    print(f"男子7車: DNF(落車/失格/欠 代理)明け × 成績（残差=通算平均着−当該着, +=良化）")
    print(f"DNF総数(7車): {dnf_total:,}\n")
    print(f"   {'区分':<14}{'n':>9}{'1着率':>8}{'top3率':>8}{'残差':>8}")
    for key in ["DNF無し(基準)", "①明け1走目", "②2走目", "③3走目", "④4-6走目", "⑤7走目以降"]:
        d = agg.get(key)
        if not d or d["n"] == 0:
            continue
        n = d["n"]
        print(f"   {key:<14}{n:>9}{d['win']/n*100:>7.1f}%{d['top3']/n*100:>7.1f}%{d['res']/n:>+8.2f}")
    print("\n※明け1-3走目の残差が基準より明確に低ければ『落車明けの尾引き』あり→落車固有取得へ。"
          "平坦なら失格/欠混入込みで信号無し。")


if __name__ == "__main__":
    main()
