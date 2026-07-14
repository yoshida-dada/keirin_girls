"""オッズ時系列スナップショット収集ランナー（S1）。

当日のガールズ開催を発見し、締切60分前〜締切の窓に入っているレースのオッズを1回取得して
SQLite（暫定）に蓄積する。cron/APScheduler から5〜10分ごとに起動する想定。
本番はPostgreSQLに置換（db/repository.py のコメント参照）。

  # 5分ごと（例, crontab）:
  # */5 * * * * cd /path/KEIRIN && python scripts/collect_odds_snapshot.py --db data/odds.sqlite

  # 単発・窓判定なしで全レース取得（手動バックフィル）:
  python scripts/collect_odds_snapshot.py --all --db data/odds.sqlite
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from db.repository import SnapshotRepo
from src.collect.gamboo_schedule import fetch_girls_kaisai, fetch_race_numbers
from src.collect.snapshot import collect_race_snapshot


def run(target: date, db_path: Path, only_in_window: bool, lead_min: int = 60) -> None:
    repo = SnapshotRepo(db_path)
    kaisai_list = fetch_girls_kaisai(target.year, target.month, target.day)
    print(f"[{datetime.now():%H:%M}] ガールズ開催 {len(kaisai_list)}件")
    total_saved = 0
    for k in kaisai_list:
        for race_no in fetch_race_numbers(k):
            try:
                r = collect_race_snapshot(k, race_no, repo,
                                          only_in_window=only_in_window, lead_min=lead_min)
            except Exception as e:                      # 個別レース失敗は握って続行
                print(f"  {k.venue_code} R{race_no}: 取得失敗 {e}")
                continue
            total_saved += r["saved"]
            if r["saved"]:
                flag = "保存"
            elif not r["is_girls"]:
                flag = "非L級"
            elif only_in_window and not r["in_window"]:
                flag = "窓外"
            else:
                flag = "skip"
            miss = f" 欠損{len(r['missing'])}" if r["missing"] else ""
            print(f"  {k.venue_code} R{race_no}: {r['n_odds']}点 締切{r['deadline']} [{flag}]{miss}")
    repo.close()
    print(f"合計 {total_saved} 行を保存: {db_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="ガールズケイリン オッズ時系列収集")
    ap.add_argument("--date", help="YYYY-MM-DD（既定=今日）")
    ap.add_argument("--db", default=str(DATA_DIR / "odds.sqlite"), help="SQLite保存先")
    ap.add_argument("--all", action="store_true",
                    help="窓判定せず全レース取得（手動バックフィル用）")
    ap.add_argument("--lead-min", type=int, default=60, help="締切何分前から集めるか")
    args = ap.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    run(target, db_path, only_in_window=not args.all, lead_min=args.lead_min)


if __name__ == "__main__":
    main()
