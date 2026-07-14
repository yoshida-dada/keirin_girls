"""プロジェクト共通設定。パスは相対解決し、秘匿値は環境変数から読む。

girls_keirin_ai_spec.md / db/schema.sql 準拠。ハードコード絶対パスは禁止。
"""
from __future__ import annotations

import os
from pathlib import Path

# --- パス ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
SCHEMA_SQL = PROJECT_ROOT / "db" / "schema.sql"

# --- 競技定数 ---
FIELD_SIZE = 7                    # 7車立て固定
TRIFECTA_COMBOS = 7 * 6 * 5       # = 210 通り
TAKEOUT_RATE = 0.25               # 3連単控除率（課題A: EVの歪みは >33% で発生）

# --- EV / 資金管理（課題F, 既定値。Phase4で狙い目マップ確定後に上書き） ---
KELLY_FRACTION = 0.25             # quarter Kelly
EV_THRESHOLDS = (1.10, 1.15, 1.20)  # バケット別チューニング対象（Phase4）
BET_UNIT_YEN = 100

# ハードリミット（円/点数, 課題F）。超過分は購入せずアラートのみ。既定値はPhase4後に調整。
BET_HARD_LIMITS = {
    "max_points_per_race":    30,      # 1レースの最大購入点数（210点中の上限）
    "max_stake_per_bet_yen":  3000,    # 1点あたり上限
    "max_stake_per_race_yen": 3000,    # 1レース合計上限（=予算）
    "max_stake_per_day_yen":  30000,   # 1日合計上限
    "max_consecutive_losses": 15,      # 連敗でkill switch（当日残りを停止）
}

# 尾部ガード（三連単のみ）。favorite-longshot bias(課題A)で極端な穴のEVが跳ねるのを抑える。
#   min_prob : これ未満の的中確率は「ノイズ穴」として購入対象外
#   max_odds : これ超の高配当は尾部不確実性が支配的なので除外（Phase4のバケット定義で見直す）
#   shrink_to_market : モデル確率 p を市場implied確率 q へ縮小 p'=(1-α)p+αq（0で無効）
EV_GUARD = {
    "min_prob": 0.005,
    "max_odds": 500.0,
    "shrink_to_market": 0.5,
}

# オッズ帯バケットの境界（課題A/Phase4。サンプル数確保のため粒度は後で調整）。
# 例 [10,30,50,100,300] → "0-10" "10-30" "30-50" "50-100" "100-300" "300+"
ODDS_BUCKET_EDGES = [10, 30, 50, 100, 300]

# --- DB 接続（環境変数優先） ---
DB_CONFIG = {
    "host": os.environ.get("KEIRIN_DB_HOST", "localhost"),
    "port": int(os.environ.get("KEIRIN_DB_PORT", "5432")),
    "dbname": os.environ.get("KEIRIN_DB_NAME", "girls_keirin"),
    "user": os.environ.get("KEIRIN_DB_USER", "keirin"),
    "password": os.environ.get("KEIRIN_DB_PASSWORD", ""),
}

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# --- スクレイピング ---
SCRAPE_MIN_INTERVAL_SEC = 0.5     # ホスト間の最小取得間隔(秒)。base.set_default_interval で実行時変更可
