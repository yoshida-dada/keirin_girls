"""並び予想（記者の隊列予想）由来の位置取り特徴。narabiテーブルを読む。

並び予想は発走前に確定している事前情報（＝as-of・リーク無し）。「誰が先頭(主導権)を打つ予定か、
誰が番手(マーク)につく予定か」を数値化する。実際に主導権を取ったかは結果の S/B(results.sb)で
分かるので、事前(並び予想)×事後(S/B)の突き合わせは analyze_narabi 側で行う。

per (race_id, car_number):
  narabi_pos  : 予想隊列位置(0=先頭, 大きいほど後方)。前ほど主導権を取りやすい位置取り。
  narabi_lead : 予想先頭(position==0)なら1、他0。
  narabi_leg  : 脚質の前がかり度（先行/押え先=2, 自在=1, 追込/差し/マーク=0）。位置取りの意図。
返り値: {(race_id, car_number): {上記3キー}}。並び予想が無いレースは含まれない。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# 脚質→前がかり度（主導権を取りに行く意図の強さ）
LEG_AGGR = {"先行": 2, "押え先": 2, "捲り": 2, "自在": 1,
            "追込": 0, "差し": 0, "マーク": 0, "追": 0}


def compute_narabi_features(db_path: str | Path) -> dict[tuple[str, int], dict]:
    """narabiテーブルから位置取り特徴を返す。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rows = conn.execute(
            "SELECT race_id, car_number, position, leg FROM narabi").fetchall()
    finally:
        conn.close()
    out: dict[tuple[str, int], dict] = {}
    for rid, car, pos, leg in rows:
        out[(rid, car)] = {
            "narabi_pos": float(pos),
            "narabi_lead": 1.0 if pos == 0 else 0.0,
            "narabi_leg": float(LEG_AGGR.get(leg, 1)),
        }
    return out


NARABI_KEYS = ["narabi_pos", "narabi_lead", "narabi_leg"]
