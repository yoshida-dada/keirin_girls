"""過去レースの並び予想をバックフィル（オッズページ再取得→parse_narabi→narabiテーブル）。

過去のオッズページにも並び予想が残っているので、本DBのガールズレース（直近 --days 日）のうち
まだ narabi が無いものを再取得して保存する。これで「並び予想(事前)× S/B(事後・既存)」の
突き合わせ検証が過去データで即可能になる（リアルタイムオッズと違いバックフィルできる）。

  python scripts/backfill_narabi.py --days 200
1レース1フェッチ・1秒間隔。既に narabi があるレースはスキップ（中断しても再開可能）。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_odds import build_odds_url
from src.collect.gamboo_racecard import parse_narabi
from src.features.rider_history import _race_date_from_id
from db.repository import DatasetRepo


def main() -> None:
    ap = argparse.ArgumentParser(description="並び予想の過去バックフィル")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--days", type=int, default=200, help="直近何日分のガールズを対象にするか")
    ap.add_argument("--limit", type=int, default=0, help="最大件数(0=無制限・動作確認用)")
    args = ap.parse_args()

    cutoff = (date.today() - timedelta(days=args.days)).isoformat()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    girls = [r[0] for r in conn.execute("SELECT race_id FROM races WHERE is_girls=1")]
    have = {r[0] for r in conn.execute("SELECT DISTINCT race_id FROM narabi")}
    conn.close()
    targets = [rid for rid in girls
               if (_race_date_from_id(rid) or "") >= cutoff and rid not in have]
    targets.sort(reverse=True)                      # 新しい順（最近を優先）
    if args.limit:
        targets = targets[:args.limit]
    print(f"対象 {len(targets)}レース（直近{args.days}日・narabi未取得）\n")

    set_default_interval(1.0)                        # 規約遵守: 1秒以上間隔
    repo = DatasetRepo(args.db)
    saved = fails = 0
    try:
        for i, rid in enumerate(targets):
            day_code, race_no, kaisai_code = rid[:14], int(rid[14:16]), rid[:10]
            try:
                html = fetch(build_odds_url(kaisai_code, day_code, race_no)).text
                n = parse_narabi(html)
                if n.get("order"):
                    repo.save_narabi(rid, n)
                    saved += 1
                else:
                    fails += 1
            except Exception as e:
                fails += 1
                if fails <= 5:
                    print(f"  {rid} 失敗: {e}")
            if (i + 1) % 100 == 0:
                print(f"  ...{i+1}/{len(targets)}  保存{saved} 失敗{fails}")
    finally:
        repo.close()
    print(f"\nバックフィル完了: 保存{saved} / 失敗{fails} / 対象{len(targets)}")


if __name__ == "__main__":
    main()
