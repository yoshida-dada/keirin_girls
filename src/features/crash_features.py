"""落車明けの表示用（バッジ）。dnf_status.status='落' から各選手の「直近落車からの経過レース数」。

予測特徴としては walk-forward で純増なし（recent_form/Elo が既に吸収）＝不採用。表示のみ。
記述統計では落車明け1走目 残差-0.34・top3 -6.9pt と実在し、6走目あたりで回復。バッジは
「落車明け N走目」を明け1〜6走目に出す（N=since+1）。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from functools import lru_cache


def _pd(s):
    try:
        return date.fromisoformat(str(s))
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=4)
def races_since_crash(db_path) -> dict:
    """{氏名: 直近落車からの経過レース数(since)}。since=0 は前走が落車＝次走が明け1走目。

    落車していない/dnf_status が無い選手はキー無し。表示側は since<=5（明け1〜6走目）で出す。
    """
    import sqlite3
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    rdt = {rid: d for rid, d in c.execute("SELECT race_id,race_date FROM races")}
    byrider = defaultdict(list)          # name -> [(date, rid)]
    for rid, nm in c.execute("SELECT race_id,rider_name FROM results WHERE rider_name IS NOT NULL"):
        d = _pd(rdt.get(rid))
        if d:
            byrider[nm].append((d, rid))
    crash = defaultdict(set)             # name -> {落車した race_id}
    try:
        for rid, nm in c.execute("SELECT race_id,rider_name FROM dnf_status WHERE status='落'"):
            crash[nm].add(rid)
    except sqlite3.OperationalError:
        c.close()
        return {}
    c.close()

    out = {}
    for nm, cset in crash.items():
        races = sorted(byrider.get(nm, []))
        idx = [i for i, (_, rid) in enumerate(races) if rid in cset]
        if not idx:
            continue
        out[nm] = len(races) - 1 - max(idx)   # 直近落車より後のレース数
    return out
