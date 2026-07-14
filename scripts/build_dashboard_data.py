"""dashboard/data.json をDBの実データ集計から生成する（S6 運用UI）。

- `data/keirin.sqlite` を **読み取り専用** で短時間だけ開いて集計し、すぐ閉じる
  （DBは収集プロセスが書き込み中のためロックを避ける）。
- 実データで埋めるのは「データ収集状況」(`data_status`) のみ。
  モデル依存の項目（推奨買い目・バケットROI・レースタイプ分布・キャリブレーション・
  累積ROI）は現時点では **未算出（status="pending"）** のプレースホルダにする。
  モデル・バックテスト完成後にこのスクリプトを拡張してそれらを実データ化する。

パスは pathlib + 相対のみ（絶対パス埋め込み禁止）。

使い方:
    python scripts/build_dashboard_data.py
    python scripts/build_dashboard_data.py --db path/to.sqlite --out path/to/data.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

# scripts/ の1つ上 = プロジェクトルート
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "keirin.sqlite"
DEFAULT_OUT = ROOT / "dashboard" / "data.json"

# バケット定義（レースタイプ×オッズ帯）。モデル完成前はスケルトンのみ。
RACE_TYPES = ["軸堅", "標準", "混戦"]
ODDS_BANDS = ["1-9倍", "10-29倍", "30-99倍", "100-299倍", "300倍+"]


def _open_ro(db_path: Path) -> sqlite3.Connection:
    """読み取り専用でDBを開く（書き込み中でもロックしない）。"""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    # 収集プロセスの書き込みを妨げない
    conn.execute("PRAGMA query_only = 1")
    return conn


def _scalar(conn: sqlite3.Connection, sql: str, default=0):
    row = conn.execute(sql).fetchone()
    return default if row is None or row[0] is None else row[0]


def collect_data_status(conn: sqlite3.Connection) -> dict:
    """DBから「データ収集状況」を集計する（実データ）。"""
    races_total = _scalar(conn, "SELECT COUNT(*) FROM races")
    d_from = _scalar(conn, "SELECT MIN(race_date) FROM races", None)
    d_to = _scalar(conn, "SELECT MAX(race_date) FROM races", None)
    race_days = _scalar(conn, "SELECT COUNT(DISTINCT race_date) FROM races")
    venues = _scalar(conn, "SELECT COUNT(DISTINCT venue_code) FROM races")

    races_with_odds = _scalar(conn, "SELECT COUNT(DISTINCT race_id) FROM odds_final_trifecta")
    races_with_results = _scalar(conn, "SELECT COUNT(DISTINCT race_id) FROM results")
    races_with_payout = _scalar(conn, "SELECT COUNT(DISTINCT race_id) FROM payouts_trifecta")

    odds_points_total = _scalar(conn, "SELECT COUNT(*) FROM odds_final_trifecta")
    entries_n = _scalar(conn, "SELECT COUNT(*) FROM entries")
    results_n = _scalar(conn, "SELECT COUNT(*) FROM results")
    payouts_n = _scalar(conn, "SELECT COUNT(*) FROM payouts_trifecta")

    field_size_dist = [
        {"size": size, "n": n}
        for (size, n) in conn.execute(
            "SELECT field_size, COUNT(*) FROM races "
            "WHERE field_size IS NOT NULL GROUP BY field_size ORDER BY field_size DESC"
        ).fetchall()
    ]

    coverage = round(races_with_payout / races_total, 3) if races_total else None
    per_race = round(odds_points_total / races_with_odds, 1) if races_with_odds else None

    return {
        "status": "ok",
        "period": {"from": d_from, "to": d_to, "race_days": race_days},
        "races_total": races_total,
        "races_with_odds": races_with_odds,
        "races_with_results": races_with_results,
        "races_with_payout": races_with_payout,
        "payout_coverage": coverage,
        "odds_points_total": odds_points_total,
        "odds_points_per_race_avg": per_race,
        "venues": venues,
        "field_size_dist": field_size_dist,
        "sample_counts": {
            "trifecta_odds_points": odds_points_total,
            "trifecta_payouts": payouts_n,
            "entries": entries_n,
            "results": results_n,
        },
    }


def pending_model_sections() -> dict:
    """モデル未完成時のプレースホルダ（未算出）。スキーマは data.json と一致。"""
    return {
        "recommendations": {
            "status": "pending",
            "date": None,
            "ev_threshold": None,
            "note": "モデル未学習のため推奨買い目は未算出です。",
            "bets": [],
        },
        "buckets": {
            "status": "pending",
            "note": "バケット分析は未算出です（Phase4で実データ化）。",
            "race_types": RACE_TYPES,
            "odds_bands": ODDS_BANDS,
            "cells": [
                {"race_type": rt, "odds_band": ob, "n": 0,
                 "roi": None, "stake": 0, "return": 0, "hits": 0}
                for rt in RACE_TYPES for ob in ODDS_BANDS
            ],
        },
        "race_type_dist": {
            "status": "pending",
            "note": "レースタイプ分類は未算出です。",
            "counts": [{"type": rt, "n": 0} for rt in RACE_TYPES],
        },
        "calibration": {
            "status": "pending",
            "note": "キャリブレーション検証は未算出です。",
            "brier": None,
            "bins": [],
        },
        "cumulative_roi": {
            "status": "pending",
            "note": "累積ROIは未算出です。",
            "final_roi": None,
            "total_stake": 0,
            "total_return": 0,
            "points": [],
        },
    }


def build(db_path: Path) -> dict:
    conn = _open_ro(db_path)
    try:
        data_status = collect_data_status(conn)
    finally:
        conn.close()  # ロックを避けるため即座に閉じる

    doc = {
        "generated": datetime.now().replace(microsecond=0).isoformat(),
        "schema_version": 1,
        "model_ready": False,
        "notice": "モデル未学習のため、推奨買い目・バケットROI・レースタイプ分布・"
                  "キャリブレーション・累積ROIは未算出です。データ収集状況のみ実データ集計です。",
        "data_status": data_status,
    }
    doc.update(pending_model_sections())
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description="dashboard/data.json をDBから生成")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="keirin.sqlite のパス")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="出力する data.json のパス")
    args = ap.parse_args()

    db_path = args.db if args.db.is_absolute() else (Path.cwd() / args.db)
    if not db_path.exists():
        raise SystemExit(f"DBが見つかりません: {db_path}")

    doc = build(db_path)

    out = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    ds = doc["data_status"]
    print(f"生成: {out}")
    print(f"  収集レース {ds['races_total']} / 期間 {ds['period']['from']}〜{ds['period']['to']}"
          f" / 払戻カバレッジ {ds['payout_coverage']}")
    print("  モデル依存項目は status='pending'（未算出）で出力しました。")


if __name__ == "__main__":
    main()
