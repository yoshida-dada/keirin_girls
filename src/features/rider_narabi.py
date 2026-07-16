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


def narabi_from_order(order: list, legs: dict) -> dict[int, dict]:
    """parse_narabi の {order:[車番...], legs:{車番:脚質}} → {車番: 生特徴}（推論時に使う）。"""
    out: dict[int, dict] = {}
    for pos, car in enumerate(order or []):
        out[car] = {
            "narabi_pos": float(pos),
            "narabi_lead": 1.0 if pos == 0 else 0.0,
            "narabi_leg": float(LEG_AGGR.get((legs or {}).get(car), 1)),
        }
    return out


def narabi_columns(cars: list[int], per_car: dict[int, dict]) -> dict[int, list]:
    """出走車 cars と各車の生narabi特徴 → モデル入力3列を車番キーで返す（学習・推論で同一）。

    narabi_pos / narabi_leg はレース内相対化（value − present平均, 欠損0）、narabi_lead は0/1のまま。
    順序は NARABI_KEYS。analyze_narabi の add_narabi と同型（train/inference skew防止）。
    """
    def rel(key):
        vals = [per_car.get(c, {}).get(key) for c in cars]
        present = [v for v in vals if v is not None]
        m = sum(present) / len(present) if present else 0.0
        return [(v - m) if v is not None else 0.0 for v in vals]

    pos_rel, leg_rel = rel("narabi_pos"), rel("narabi_leg")
    out: dict[int, list] = {}
    for i, c in enumerate(cars):
        lead = per_car.get(c, {}).get("narabi_lead")
        out[c] = [pos_rel[i], float(lead) if lead is not None else 0.0, leg_rel[i]]
    return out
