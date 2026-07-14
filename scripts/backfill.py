"""過去ガールズ戦の一括収集バックフィル（S1・学習＋バックテスト用データセット）。

指定期間の各日のガールズ開催を辿り、L級レースの 出走表／着順／三連単払戻／確定オッズ全210点 を
SQLite（無料運用）に蓄積する。収集済みレースはスキップして再開可能。

  # 直近30日を収集:
  python scripts/backfill.py --days 30 --db data/keirin.sqlite
  # 期間指定 & 動作確認用に最大5レースで打ち切り:
  python scripts/backfill.py --date-from 2026-06-14 --date-to 2026-07-13 --max-races 5
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from db.repository import DatasetRepo
from src.collect.dataset import collect_race_dataset
from src.collect.gamboo_schedule import (
    fetch_girls_kaisai, fetch_girls_race_numbers, kaisai_race_date,
)
from src.collect.snapshot import build_race_id


def _daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def _persist(repo: DatasetRepo, ds, day: date) -> None:
    """1レース分の収集結果をDBへ（メインスレッドのみが呼ぶ＝SQLite安全）。"""
    repo.save_race(ds.race_id, day.isoformat(), ds.venue_code, ds.race_no,
                   ds.is_girls, ds.deadline, ds.field_size)
    repo.save_entries(ds.race_id, ds.entries)
    repo.save_recent_form(ds.race_id, ds.recent)
    repo.save_results(ds.race_id, ds.results)
    repo.save_odds_final(ds.race_id, ds.odds_final)
    repo.save_payout(ds.race_id, ds.payout)


def run(d_from: date, d_to: date, db_path: Path, max_races: int | None,
        workers: int = 1) -> None:
    """並列収集: オッズ/結果ページの取得＋パースを workers 本で並行し、DB書き込みは直列。"""
    repo = DatasetRepo(db_path)
    done = set(repo.race_ids())
    saved = skipped = men = errors = 0

    for day in _daterange(d_from, d_to):
        try:
            kaisai_list = fetch_girls_kaisai(day.year, day.month, day.day)
        except Exception as e:
            print(f"{day} 開催一覧取得失敗: {e}")
            continue
        # 開催一覧は初日〜最終日の全日程を含むため、実施日が当日と一致する開催のみに絞る
        # （初日/2日目の混在を排除し、race_dateを実施日として正しく保存する）。
        kaisai_list = [k for k in kaisai_list if kaisai_race_date(k.kaisai_day_code) == day]
        if not kaisai_list:
            continue
        # 当日の未収集タスク (kaisai, race_no) を集める。レース一覧ページの級班列から
        # ガールズ(L級)レースだけを直接絞り込むため、男子戦のオッズページは一切取得しない。
        tasks = []
        for k in kaisai_list:
            try:
                girls_nos = fetch_girls_race_numbers(k)
            except Exception as e:
                errors += 1
                print(f"  {k.venue_code} レース一覧失敗: {e}")
                continue
            for rno in girls_nos:
                rid = build_race_id(k.kaisai_day_code, rno)
                if rid in done:
                    skipped += 1
                else:
                    tasks.append((k, rno))
        if not tasks:
            continue
        print(f"{day}: ガールズ開催{len(kaisai_list)}件 / 収集対象{len(tasks)}レース (workers={workers})")

        day_saved = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(collect_race_dataset, k, rno, True): (k, rno) for k, rno in tasks}
            for fut in as_completed(futs):
                try:
                    ds = fut.result()
                except Exception as e:
                    errors += 1
                    k, rno = futs[fut]
                    print(f"  {k.venue_code} R{rno}: 取得失敗 {e}")
                    continue
                if not ds.is_girls:
                    men += 1
                    continue
                _persist(repo, ds, day)
                done.add(ds.race_id)
                saved += 1
                day_saved += 1
                if max_races is not None and saved >= max_races:
                    break
        print(f"  → {day} 保存{day_saved}（累計{saved}）")
        if max_races is not None and saved >= max_races:
            print(f"max-races({max_races})到達で打ち切り")
            break
    _summary(repo, saved, skipped, men, errors)


def _summary(repo: DatasetRepo, saved: int, skipped: int, men: int, errors: int) -> None:
    print(f"\n保存{saved} / スキップ済{skipped} / 非L級{men} / 失敗{errors}")
    print(f"DB累計: races={repo.count('races')} entries={repo.count('entries')} "
          f"results={repo.count('results')} odds={repo.count('odds_final_trifecta')} "
          f"payouts={repo.count('payouts_trifecta')}")
    repo.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="ガールズ戦 過去データ一括収集")
    ap.add_argument("--days", type=int, default=30, help="今日から遡る日数（既定30）")
    ap.add_argument("--date-from"); ap.add_argument("--date-to")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--max-races", type=int, help="動作確認用の打ち切り上限")
    ap.add_argument("--interval", type=float, help="ホスト間の取得間隔(秒)。既定はsettings値")
    ap.add_argument("--workers", type=int, default=1,
                    help="並列フェッチ数（>1でオッズ/結果ページを同時取得。上限8）")
    args = ap.parse_args()

    from src.collect.base import set_default_interval
    workers = max(1, min(8, args.workers))
    if args.interval is not None:
        set_default_interval(args.interval)
        print(f"取得間隔 {args.interval}s")
    elif workers > 1:
        # 並列時はレートを同時実行数で制御するためホスト間隔を0に（ネットワーク律速）。
        set_default_interval(0.0)

    d_to = date.fromisoformat(args.date_to) if args.date_to else date.today()
    d_from = date.fromisoformat(args.date_from) if args.date_from else d_to - timedelta(days=args.days)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"収集期間 {d_from} 〜 {d_to} → {db_path} (workers={workers})")
    run(d_from, d_to, db_path, args.max_races, workers=workers)


if __name__ == "__main__":
    main()
