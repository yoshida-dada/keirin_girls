"""出走間隔（前走からの日数）特徴（男子で採用）。

walk-forward検証（scripts/validate_interval_wf.py / validate_interval_e1.py）で、E1(線形・120日上限)
が全体top1 +0.36pt(5/5fold一貫)・ECE悪化なし・長期ブランク在籍レースでtop1+2pt級と純増。
効果は「長期斡旋切れ/故障明けの選手を過大評価しない」点に集中（紐tri10は中立、1着精度向上）。

エンコードは学習・推論で同一関数(interval_columns)を通す＝train/inference skew防止。
  gap_lin = min(gap_days, 120) / 30    （gap不明は0＝直近に走った扱い＝ペナルティ無しの良性既定）

as-of厳守: gap は「発走前に既知」の情報（前走日から当日までの経過日数）のみを使う（リーク無し）。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

INTERVAL_KEYS = ["gap_lin"]

_GAP_CAP = 120
_GAP_SCALE = 30.0


def encode_gap(gap_days: float | None) -> float:
    """前走からの日数 → 特徴値。不明(None)は0（＝直近に走った扱い・ペナルティ無し）。"""
    if gap_days is None or gap_days <= 0:
        return 0.0
    return min(gap_days, _GAP_CAP) / _GAP_SCALE


def interval_columns(car_numbers, gap_by_car: dict) -> dict:
    """{車番: [gap_lin]} を返す（学習・推論共通）。gap_by_car は {車番: 前走からの日数}。"""
    return {c: [encode_gap(gap_by_car.get(c))] for c in car_numbers}


def _race_date(rid: str) -> date:
    # race_id = [場2][初日YYYYMMDD8][開催日目NN2][R...]。実施日 = 初日 + (NN-1)。
    return date(int(rid[2:6]), int(rid[6:8]), int(rid[8:10])) + timedelta(days=int(rid[10:12]) - 1)


def compute_pre_race_gap(db_path) -> dict:
    """学習・評価用: {(race_id, car_number): 前走からの日数} を **完走履歴(results)** から as-of で算出。

    推論側の日数(days_since)は current_stats が results(着順あり)JOIN races.race_date で出すため、
    学習も同じ results ベースにして train/inference skew を無くす（欠場は前走に数えない）。
    各選手(氏名)の完走日を時系列に並べ、各レースに「直前の完走日からの経過日数」を付ける
    （初出走はキー無し＝推論側で0扱い）。
    """
    import sqlite3
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    rows = c.execute(
        "SELECT ra.race_id, res.car_number, res.rider_name, ra.race_date"
        " FROM results res JOIN races ra ON res.race_id = ra.race_id"
        " WHERE res.position IS NOT NULL AND res.rider_name IS NOT NULL").fetchall()
    c.close()
    from datetime import date as _d
    def _pd(s):
        try:
            return _d.fromisoformat(str(s))
        except (ValueError, TypeError):
            return None
    byname = defaultdict(set)
    for rid, car, nm, rdate in rows:
        d = _pd(rdate)
        if d:
            byname[nm].add((d, rid))
    gap_by_rn = {}
    for nm, s in byname.items():
        prev = None
        for d, rid in sorted(s):
            if prev is not None and (d - prev).days > 0:
                gap_by_rn[(rid, nm)] = (d - prev).days
            prev = d
    out = {}
    for rid, car, nm, _ in rows:
        g = gap_by_rn.get((rid, nm))
        if g is not None:
            out[(rid, car)] = g
    return out
