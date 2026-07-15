"""選手のas-ofローリング成績特徴量（研究用）。

収集済みの races/entries/results を**日付順**に処理し、各選手（氏名で同定）の履歴を
更新しながら、各エントリの**発走前(as-of)**成績を返す。Eloと同じく「そのレースの結果は
特徴量に含めず、特徴量を記録した後に履歴を更新する」ことでリークを防ぐ（src/model/elo.py
の compute_pre_race_elo と同じ作法）。選手は登録番号が無いため氏名で同定する。

返す発走前特徴（各エントリ = (race_id, car_number) 単位）:
  career_win_rate   : これまでの通算1着率（1着数 / 出走数）。履歴無しは None
  recent5_avg_finish: 直近5走の平均着順。履歴無しは None
  venue_win_rate    : 当該競輪場(venue_code)での通算1着率。当該場の履歴無しは None
  days_since_last   : 前走からの経過日数。初出走は None
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from datetime import date
from pathlib import Path


def compute_rolling(db_path: str | Path) -> dict[tuple[str, int], dict]:
    """各エントリ(race_id, car_number)の**発走前**ローリング成績を返す。

    レースを日付順(race_date, race_id)に処理し、各選手(氏名)の履歴を更新しながら、
    処理中レースについては「更新前(=発走前)」の集計値を記録する。
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        races = conn.execute(
            "SELECT race_id, race_date, venue_code FROM races"
            " ORDER BY race_date, race_id").fetchall()

        # 選手氏名ごとの as-of 履歴（発走前の集計に使う累積値のみ保持）
        starts: dict[str, int] = defaultdict(int)             # 通算出走数
        wins: dict[str, int] = defaultdict(int)               # 通算1着数
        venue_starts: dict[tuple[str, str], int] = defaultdict(int)  # (氏名,場)出走数
        venue_wins: dict[tuple[str, str], int] = defaultdict(int)    # (氏名,場)1着数
        recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))  # 直近5着順
        last_date: dict[str, date] = {}                       # 前走日

        out: dict[tuple[str, int], dict] = {}
        for race_id, race_date, venue_code in races:
            rdate = date.fromisoformat(race_date) if race_date else None
            ents = conn.execute(
                "SELECT car_number, rider_name FROM entries WHERE race_id=?",
                (race_id,)).fetchall()
            car_name = {c: n for c, n in ents}
            positions = dict(conn.execute(
                "SELECT car_number, position FROM results"
                " WHERE race_id=? AND position IS NOT NULL", (race_id,)).fetchall())

            # --- 発走前(as-of)特徴を記録（この時点の履歴は当該レースを含まない） ---
            for c, name in car_name.items():
                s = starts[name]
                vs = venue_starts[(name, venue_code)]
                rc = recent[name]
                ld = last_date.get(name)
                out[(race_id, c)] = {
                    "career_win_rate": (wins[name] / s) if s > 0 else None,
                    "recent5_avg_finish": (sum(rc) / len(rc)) if rc else None,
                    "venue_win_rate": (venue_wins[(name, venue_code)] / vs)
                                      if vs > 0 else None,
                    "days_since_last": (rdate - ld).days
                                       if (ld is not None and rdate is not None) else None,
                }

            # --- 当該レースの結果で履歴を更新（記録後に行うのでリーク無し） ---
            for c, pos in positions.items():
                name = car_name.get(c)
                if name is None:
                    continue
                starts[name] += 1
                venue_starts[(name, venue_code)] += 1
                if pos == 1:
                    wins[name] += 1
                    venue_wins[(name, venue_code)] += 1
                recent[name].append(pos)
                if rdate is not None:
                    last_date[name] = rdate

        return out
    finally:
        conn.close()
