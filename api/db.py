"""SQLite 読み取り専用アクセス（Phase5 API）。

`data/keirin.sqlite` を **mode=ro** で **短時間だけ** 開いて集計し、すぐ閉じる
（収集プロセスの書き込みロックを避ける）。data.json のデータ契約と一致させるため、
集計ロジックは既存の scripts/build_dashboard_data.py を import して再利用する（重複実装しない）。
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from api import _bootstrap  # noqa: F401  sys.path にプロジェクトルートを載せる

# scripts の集計ロジックをそのまま流用（data.json と同一スキーマを保証）
from scripts.build_dashboard_data import collect_data_status, pending_model_sections

PROJECT_ROOT = _bootstrap.PROJECT_ROOT


def default_db_path() -> Path:
    """既定のSQLiteパス。環境変数 KEIRIN_DB_SQLITE で上書き可（絶対パス埋め込み禁止）。"""
    env = os.environ.get("KEIRIN_DB_SQLITE")
    if env:
        return Path(env)
    return PROJECT_ROOT / "data" / "keirin.sqlite"


def open_ro(db_path: Path | None = None) -> sqlite3.Connection:
    """読み取り専用でDBを開く（書き込み中でもロックしない）。呼び出し側で必ず close する。"""
    path = Path(db_path) if db_path else default_db_path()
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    conn.execute("PRAGMA query_only = 1")
    return conn


def get_data_status(db_path: Path | None = None) -> dict:
    """データ収集状況（data.json の data_status 相当）を実データで返す。"""
    conn = open_ro(db_path)
    try:
        return collect_data_status(conn)
    finally:
        conn.close()  # ロックを避けるため即座に閉じる


def get_pending_sections() -> dict:
    """モデル未算出セクション（buckets / race_type_dist 等）を data.json 形で返す。"""
    return pending_model_sections()
