"""S0 基盤スモークテスト: 設定・定数・スキーマの整合を確認する。"""
from pathlib import Path

from config import settings


def test_competition_constants():
    assert settings.FIELD_SIZE == 7
    assert settings.TRIFECTA_COMBOS == 210  # 7*6*5
    assert 0 < settings.TAKEOUT_RATE < 1
    assert settings.KELLY_FRACTION == 0.25  # quarter Kelly


def test_paths_are_absolute_and_rooted():
    assert settings.PROJECT_ROOT.is_absolute()
    assert settings.SCHEMA_SQL.name == "schema.sql"


def test_schema_defines_required_tables():
    sql = settings.SCHEMA_SQL.read_text(encoding="utf-8")
    # 仕様書Phase1が要求する中核テーブル + 全点オッズ要件
    for table in [
        "races", "riders", "entries", "results",
        "odds_snapshots_trifecta", "odds_final_trifecta",
    ]:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql, table


def test_db_config_keys():
    for key in ("host", "port", "dbname", "user", "password"):
        assert key in settings.DB_CONFIG
