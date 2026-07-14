"""学習データ読み込み（S3学習）。収集済みSQLiteから 特徴量行列＋着順ラベル を組む。

backfillが保存した entries/results から、各レースの特徴量（assembler, recent=None）と観測着順
（上位3車）を取り出す。時系列分割のため race_date を保持する。

注意: backfillは直近4ヶ月(recent_form)・ageを保存していないため、それら由来の特徴量はNaN。
本モジュールはレース内で変動する使用可能特徴量のみ返す（PLでは race一定の特徴量は打ち消されるため）。
recent_form の付与は後日のデータ拡張（docs/design_s2_features.md）。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.collect.gamboo_racecard import Entry, RecentForm
from src.features.assembler import build_features

# PLに効く「レース内変動」特徴量（race一定の broadcast 特徴量は除外。PLはシフト不変）。
# 基本セット（entriesのみで常に得られる）。
PL_FEATURES = [
    "racing_score", "gear_ratio", "rel_score_max", "score_rank",
    "is_escape", "is_dash", "is_closing", "is_mark",
]
# 拡張セット（recent_form/age 収集後。直近の調子・脚質率・派生を追加）。
PL_FEATURES_FULL = PL_FEATURES + [
    "age", "win_rate", "top3_rate",
    "escape_rate", "dash_rate", "closing_rate", "mark_rate", "b_rate",
    "recent_avg_finish", "stability", "escape_ev", "tenkai_advantage",
]

_ENTRY_COLS = ["car_number", "bracket_number", "rider_name", "prefecture", "age",
               "term", "class_rank", "leg_type", "gear_ratio", "racing_score"]
_RECENT_COLS = ["car_number", "s_count", "b_count", "escape_cnt", "dash_cnt", "closing_cnt",
                "mark_cnt", "first_cnt", "second_cnt", "third_cnt", "out_cnt",
                "win_rate", "top2_rate", "top3_rate"]


@dataclass
class RaceSample:
    race_id: str
    date: str
    car_numbers: list[int]      # X の行順に対応する車番
    X: np.ndarray               # (n_riders, n_features)
    order: list[int]            # 着順どおりの車番（1着,2着,3着）
    feature_names: list[str]


def _entries_of(conn, race_id: str) -> list[Entry]:
    rows = conn.execute(
        f"SELECT {','.join(_ENTRY_COLS)} FROM entries WHERE race_id=? ORDER BY car_number",
        (race_id,)).fetchall()
    return [Entry(car_number=r[0], bracket_number=r[1], rider_name=r[2], prefecture=r[3],
                  age=r[4], term=r[5], class_rank=r[6], leg_type=r[7],
                  gear_ratio=r[8], racing_score=r[9]) for r in rows]


def _recent_of(conn, race_id: str) -> dict[int, RecentForm]:
    """recent_form テーブルから {車番: RecentForm} を復元（無ければ空＝旧DB互換）。"""
    try:
        rows = conn.execute(
            f"SELECT {','.join(_RECENT_COLS)} FROM recent_form WHERE race_id=?",
            (race_id,)).fetchall()
    except sqlite3.OperationalError:
        return {}
    out = {}
    for r in rows:
        out[r[0]] = RecentForm(
            car_number=r[0], s_count=r[1], b_count=r[2], escape=r[3], dash=r[4],
            closing=r[5], mark=r[6], first=r[7], second=r[8], third=r[9], out=r[10],
            win_rate=r[11], top2_rate=r[12], top3_rate=r[13])
    return out


def _order_of(conn, race_id: str) -> list[int]:
    rows = conn.execute(
        "SELECT position, car_number FROM results WHERE race_id=? AND position IS NOT NULL"
        " ORDER BY position", (race_id,)).fetchall()
    return [car for _, car in rows]


def load_samples(db_path: str | Path, field_size: int = 7,
                 features: list[str] = PL_FEATURES) -> list[RaceSample]:
    """結果のある field_size 車レースを RaceSample のリストで返す（date昇順）。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        races = conn.execute(
            "SELECT race_id, race_date FROM races WHERE field_size=? ORDER BY race_date, race_id",
            (field_size,)).fetchall()
        samples: list[RaceSample] = []
        for race_id, date in races:
            order = _order_of(conn, race_id)
            if len(order) < 3:
                continue                      # 上位3着が確定しないレースは学習に使わない
            entries = _entries_of(conn, race_id)
            if len(entries) != field_size:
                continue
            recent = _recent_of(conn, race_id)
            df = build_features(entries, recent)   # recentがあれば直近4ヶ月特徴も入る
            if df[features].isna().any().any():
                continue                      # 要求特徴に欠損があればスキップ
            cars = list(df.index)
            X = df.loc[cars, features].to_numpy(dtype=float)
            # order の車番が全て出走表に含まれることを確認
            if not set(order[:3]).issubset(set(cars)):
                continue
            samples.append(RaceSample(race_id=race_id, date=date, car_numbers=cars,
                                      X=X, order=order[:3], feature_names=list(features)))
        return samples
    finally:
        conn.close()


def standardize(samples: list[RaceSample]) -> tuple[np.ndarray, np.ndarray]:
    """全リーダー行を縦に積んで各特徴量の mean/std を返す（学習時の標準化用）。"""
    allX = np.vstack([s.X for s in samples]) if samples else np.zeros((0, len(PL_FEATURES)))
    mean = allX.mean(axis=0)
    std = allX.std(axis=0)
    std[std == 0] = 1.0
    return mean, std
