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
    field_size  INTEGER,
    grade       TEXT,      -- G1/G2/G3/F1/F2/GP。バケット分析で格別に層別するのに要る
    race_name   TEXT       -- "Ｓ級決勝"/"二次予選" 等の生文字列（正規化は表示側）
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
    race_id     TEXT NOT NULL,
    car_number  INTEGER NOT NULL,
    position    INTEGER NOT NULL,  -- 記者の並び予想の隊列位置(0=先頭。全ライン通しの通し番号)
    leg         TEXT,              -- 脚質(先行/自在/追込/押え先 等)
    line_id     INTEGER,           -- 何本目のラインか(0起点)。単騎も1本として数える
    pos_in_line INTEGER,           -- ライン内の位置(0=ライン先頭, 1=番手, 2=3番手…)
    seri_group  INTEGER,           -- 競り(同じ位置を争うグループ)の通し番号。競り以外はNULL
    PRIMARY KEY (race_id, car_number)
);
"""


class DatasetRepo:
    """学習＋バックテスト用データセットのSQLiteリポジトリ（races/entries/results/確定オッズ/払戻）。"""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.executescript(_DATASET_DDL)
        self._add_missing_columns()

    # 後から足した列。CREATE TABLE IF NOT EXISTS は既存テーブルを変更しないので、
    # ここに書いた列は起動時に ALTER で補う。**足した列はここにも必ず追加すること**
    # （narabi の line_id/pos_in_line を書き忘れ、旧DBで INSERT の列数が合わず
    #  結果取得が丸ごと落ちた）。
    _ADDED_COLUMNS = {
        "races": (("grade", "TEXT"), ("race_name", "TEXT")),
        "narabi": (("line_id", "INTEGER"), ("pos_in_line", "INTEGER"),
                   ("seri_group", "INTEGER")),
    }

    def _add_missing_columns(self) -> None:
        """既存DBに後から足した列を補う。

        SQLite の ADD COLUMN は NULL許容なら既存行を書き換えずに済むので無停止で通る。
        """
        for table, cols in self._ADDED_COLUMNS.items():
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if not have:                      # そのテーブル自体が無い（DDL適用前）
                continue
            for col, decl in cols:
                if col not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self.conn.commit()

    def save_race(self, race_id: str, race_date: str, venue_code: str, race_no: int,
                  is_girls: bool, deadline: str | None, field_size: int,
                  grade: str | None = None, race_name: str | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO races"
            " (race_id, race_date, venue_code, race_no, is_girls, deadline, field_size,"
            "  grade, race_name)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (race_id, race_date, venue_code, race_no, int(is_girls), deadline, field_size,
             grade, race_name))
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
        """並び予想を保存。

        {"order":[車番...(前→後)], "legs":{車番:脚質}, "lines":[[車番...], ...]}
        lines が無い旧データ（ガールズ既存分）は line_id/pos_in_line を NULL にする。
        """
        order = (narabi or {}).get("order") or []
        legs = (narabi or {}).get("legs") or {}
        lines = (narabi or {}).get("lines") or []
        pos_of: dict[int, tuple[int, int]] = {}
        for li, line in enumerate(lines):
            for pi, car in enumerate(line):
                pos_of[car] = (li, pi)
        # 競り（同じ位置を争うグループ）。「連続する競り＝1グループ」では復元できない
        # （4人連続の競りは実ページ8件すべてが2グループだった）ので、パース時に必ず保存する。
        seri_of: dict[int, int] = {}
        for gi, g in enumerate((narabi or {}).get("seri") or []):
            for car in g:
                seri_of[car] = gi
        rows = [(race_id, car, pos, legs.get(car),
                 pos_of.get(car, (None, None))[0], pos_of.get(car, (None, None))[1],
                 seri_of.get(car))
                for pos, car in enumerate(order)]
        if rows:
            # 列名を明示する。VALUES(?,?,…) の位置指定だと列を足すたびに旧DBで
            # 「N columns but M values」で落ちる（実際にそれで結果取得が止まった）。
            self.conn.executemany(
                "INSERT OR REPLACE INTO narabi"
                " (race_id, car_number, position, leg, line_id, pos_in_line, seri_group)"
                " VALUES (?,?,?,?,?,?,?)", rows)
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
