"""オッズ時系列スナップショットの永続化（S1）。

暫定的にSQLiteで検証する（本番は db/schema.sql のPostgreSQL。DDLはサブセット互換）。
PostgreSQLへ移行する際は接続層だけ差し替え、combo表記/主キーはそのまま使える。
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

# schema.sql の odds_snapshots_trifecta のSQLite互換サブセット
_DDL = """
CREATE TABLE IF NOT EXISTS odds_snapshots_trifecta (
    race_id   TEXT NOT NULL,
    combo     TEXT NOT NULL,   -- 車番を "-" 連結（着順どおり 例 "3-1-5"）
    odds      REAL NOT NULL,
    taken_at  TEXT NOT NULL,   -- ISO8601
    PRIMARY KEY (race_id, combo, taken_at)
);
"""


def combo_to_str(combo: tuple[int, int, int]) -> str:
    return "-".join(str(x) for x in combo)


def combo_from_str(s: str) -> tuple[int, int, int]:
    a, b, c = (int(x) for x in s.split("-"))
    return (a, b, c)


class SnapshotRepo:
    """オッズ時系列スナップショットのSQLiteリポジトリ。"""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(_DDL)

    def save_snapshot(
        self, race_id: str, odds: dict[tuple[int, int, int], float], taken_at: datetime
    ) -> int:
        """1時点の全点オッズを保存する。同一(race,combo,時刻)は上書き。保存件数を返す。"""
        ts = taken_at.replace(microsecond=0).isoformat()
        rows = [(race_id, combo_to_str(c), float(o), ts) for c, o in odds.items()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO odds_snapshots_trifecta"
            " (race_id, combo, odds, taken_at) VALUES (?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def snapshot_times(self, race_id: str) -> list[str]:
        """あるレースで蓄積済みのスナップショット時刻一覧（昇順）。"""
        cur = self.conn.execute(
            "SELECT DISTINCT taken_at FROM odds_snapshots_trifecta"
            " WHERE race_id=? ORDER BY taken_at",
            (race_id,),
        )
        return [r[0] for r in cur.fetchall()]

    def load_snapshot(self, race_id: str, taken_at: str) -> dict[tuple[int, int, int], float]:
        """指定時刻の全点オッズを復元する。"""
        cur = self.conn.execute(
            "SELECT combo, odds FROM odds_snapshots_trifecta WHERE race_id=? AND taken_at=?",
            (race_id, taken_at),
        )
        return {combo_from_str(c): o for c, o in cur.fetchall()}

    def close(self) -> None:
        self.conn.close()


# schema.sql（PostgreSQL）のSQLite互換サブセット。学習＋バックテスト用データセット。
# 無料運用のためSQLite単一ファイル。本番PostgreSQLへは接続層のみ差し替え。
_DATASET_DDL = """
CREATE TABLE IF NOT EXISTS races (
    race_id     TEXT PRIMARY KEY,
    race_date   TEXT,
    venue_code  TEXT,
    race_no     INTEGER,
    is_girls    INTEGER,
    deadline    TEXT,
    field_size  INTEGER
);
CREATE TABLE IF NOT EXISTS entries (
    race_id       TEXT NOT NULL,
    car_number    INTEGER NOT NULL,
    bracket_number INTEGER,
    rider_name    TEXT,
    prefecture    TEXT,
    age           INTEGER,
    term          INTEGER,
    class_rank    TEXT,
    leg_type      TEXT,
    gear_ratio    REAL,
    racing_score  REAL,
    PRIMARY KEY (race_id, car_number)
);
-- 出走表同梱の直近4ヶ月成績（as-osスタッツ, S2特徴量の第一次ソース）。
CREATE TABLE IF NOT EXISTS recent_form (
    race_id     TEXT NOT NULL,
    car_number  INTEGER NOT NULL,
    s_count     INTEGER,
    b_count     INTEGER,
    escape_cnt  INTEGER,
    dash_cnt    INTEGER,
    closing_cnt INTEGER,
    mark_cnt    INTEGER,
    first_cnt   INTEGER,
    second_cnt  INTEGER,
    third_cnt   INTEGER,
    out_cnt     INTEGER,
    win_rate    REAL,
    top2_rate   REAL,
    top3_rate   REAL,
    PRIMARY KEY (race_id, car_number)
);
CREATE TABLE IF NOT EXISTS results (
    race_id     TEXT NOT NULL,
    position    INTEGER,
    car_number  INTEGER NOT NULL,
    rider_name  TEXT,
    margin      TEXT,
    last_lap    REAL,
    kimarite    TEXT,
    sb          TEXT,
    PRIMARY KEY (race_id, car_number)
);
CREATE TABLE IF NOT EXISTS odds_final_trifecta (
    race_id TEXT NOT NULL,
    combo   TEXT NOT NULL,
    odds    REAL NOT NULL,
    PRIMARY KEY (race_id, combo)
);
CREATE TABLE IF NOT EXISTS payouts_trifecta (
    race_id    TEXT NOT NULL,
    combo      TEXT NOT NULL,
    payout     INTEGER NOT NULL,
    popularity INTEGER,
    PRIMARY KEY (race_id, combo)
);
CREATE TABLE IF NOT EXISTS narabi (
    race_id    TEXT NOT NULL,
    car_number INTEGER NOT NULL,
    position   INTEGER NOT NULL,   -- 記者の並び予想の隊列位置(0=先頭)
    leg        TEXT,               -- 脚質(先行/自在/追込/押え先 等)
    PRIMARY KEY (race_id, car_number)
);
"""


class DatasetRepo:
    """学習＋バックテスト用データセットのSQLiteリポジトリ（races/entries/results/確定オッズ/払戻）。"""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(_DATASET_DDL)

    def save_race(self, race_id: str, race_date: str, venue_code: str, race_no: int,
                  is_girls: bool, deadline: str | None, field_size: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO races"
            " (race_id, race_date, venue_code, race_no, is_girls, deadline, field_size)"
            " VALUES (?,?,?,?,?,?,?)",
            (race_id, race_date, venue_code, race_no, int(is_girls), deadline, field_size))
        self.conn.commit()

    def save_entries(self, race_id: str, entries: list) -> int:
        rows = [(race_id, e.car_number, e.bracket_number, e.rider_name, e.prefecture,
                 e.age, e.term, e.class_rank, e.leg_type, e.gear_ratio, e.racing_score)
                for e in entries]
        self.conn.executemany(
            "INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def save_recent_form(self, race_id: str, recent: dict) -> int:
        """{車番: RecentForm} を保存（出走表同梱の直近4ヶ月成績）。"""
        rows = [(race_id, f.car_number, f.s_count, f.b_count, f.escape, f.dash,
                 f.closing, f.mark, f.first, f.second, f.third, f.out,
                 f.win_rate, f.top2_rate, f.top3_rate) for f in recent.values()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO recent_form VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def save_results(self, race_id: str, results: list) -> int:
        rows = [(race_id, r.position, r.car_number, r.rider_name, r.margin,
                 r.last_lap, r.kimarite, r.sb) for r in results]
        self.conn.executemany(
            "INSERT OR REPLACE INTO results VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def save_odds_final(self, race_id: str, odds: dict) -> int:
        rows = [(race_id, combo_to_str(c), float(o)) for c, o in odds.items()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO odds_final_trifecta VALUES (?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def save_narabi(self, race_id: str, narabi: dict) -> int:
        """並び予想 {"order":[車番...(前→後)], "legs":{車番:脚質}} を保存。"""
        order = (narabi or {}).get("order") or []
        legs = (narabi or {}).get("legs") or {}
        rows = [(race_id, car, pos, legs.get(car)) for pos, car in enumerate(order)]
        if rows:
            self.conn.executemany("INSERT OR REPLACE INTO narabi VALUES (?,?,?,?)", rows)
            self.conn.commit()
        return len(rows)

    def save_payout(self, race_id: str, payout) -> None:
        if payout is None:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO payouts_trifecta VALUES (?,?,?,?)",
            (race_id, combo_to_str(payout.combo), payout.payout, payout.popularity))
        self.conn.commit()

    def race_ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT race_id FROM races ORDER BY race_id")]

    def count(self, table: str) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
