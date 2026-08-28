"""落車固有(dnf_status.status='落')の明け × 成績（男子7車）。backfill_dnf_status 実行後に使う。

DNF代理(position NULL)では落車が約4割で信号が薄まっていた。ここは落車だけを特定して回復
カーブを見る。残差=通算平均着−当該着(+=良化)。落車明け何走目で基準に戻るかを測り、特徴の
減衰形(例 exp / 1..6走ランプ)を決める。

  PYTHONIOENCODING=utf-8 python scripts/analyze_crash_specific.py
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
    # 落車イベント: (race_id, rider_name)
    fell = set()
    try:
        for rid, nm in c.execute("SELECT race_id,rider_name FROM dnf_status WHERE status='落'"):
            fell.add((rid, nm))
    except sqlite3.OperationalError:
        print("dnf_status が無い。先に backfill_dnf_status.py を実行。"); return
    c.close()
    print(f"落車イベント(dnf_status='落'): {len(fell):,}")

    def pd(s):
        try:
            return date.fromisoformat(str(s))
        except (ValueError, TypeError):
            return None

    byrider = defaultdict(list)
    for rid, nm, pos in rows:
        if fs.get(rid) != 7:
            continue
        d = pd(rdt.get(rid))
        if d:
            byrider[nm].append((d, rid, pos))
    cbase = {}
    for nm, v in byrider.items():
        fin = [p for _, _, p in v if p is not None]
        if fin:
            cbase[nm] = sum(fin) / len(fin)

    agg = defaultdict(lambda: {"n": 0, "res": 0.0, "top3": 0, "win": 0})
    for nm, v in byrider.items():
        if nm not in cbase:
            continue
        v.sort()
        base = cbase[nm]
        since = None
        for d, rid, pos in v:
            crashed = (rid, nm) in fell
            if crashed:
                since = 0
                continue                 # 落車したレース自体は成績集計に入れない
            if pos is None:
                continue                 # 落車以外のDNF(失格/欠)は集計から除外
            if since is None:
                key = "落車履歴なし(基準)"
            else:
                since += 1
                key = ("①明け1走目" if since == 1 else "②2走目" if since == 2 else
                       "③3走目" if since == 3 else "④4-6走目" if since <= 6 else "⑤7走目以降")
            a = agg[key]
            a["n"] += 1; a["res"] += base - pos; a["top3"] += int(pos <= 3); a["win"] += int(pos == 1)

    print(f"\n男子7車: 落車明け × 成績（残差=通算平均着−当該着, +=良化）")
    print(f"   {'区分':<16}{'n':>9}{'1着率':>8}{'top3率':>8}{'残差':>8}")
    for key in ["落車履歴なし(基準)", "①明け1走目", "②2走目", "③3走目", "④4-6走目", "⑤7走目以降"]:
        d = agg.get(key)
        if not d or d["n"] == 0:
            continue
        n = d["n"]
        print(f"   {key:<16}{n:>9}{d['win']/n*100:>7.1f}%{d['top3']/n*100:>7.1f}%{d['res']/n:>+8.2f}")
    print("\n※DNF代理(-0.18)より落車固有で低下が深ければ特徴として強い。回復走数で減衰形を決める。")


if __name__ == "__main__":
    main()
