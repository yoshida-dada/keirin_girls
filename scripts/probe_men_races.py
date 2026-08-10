"""【Phase 0】男子競輪の試験収集。既存パーサが通るかの確認と、判断材料の収集。

**本番DB(data/keirin.sqlite)には書かない。** 方針が固まる前に男女を混ぜないため、
専用の `data/keirin_men_probe.sqlite` に保存する（men_keirin_plan.md の Phase 0 方針）。

やること:
  - 指定日の各開催について「全レース番号 − ガールズレース番号」＝男子レースを列挙
  - 既存の collect_race_dataset(require_girls=False) で1レースずつ収集
  - 出走表/直近成績/並び予想/確定オッズ/着順/払戻 をプローブDBへ保存
  - パースできなかった項目をレース単位で記録し、最後に集計して出す

スクレイピング規約（CLAUDE.md 課題G）に従い間隔は既定1.2秒。--limit で必ず上限を切る。

  PYTHONIOENCODING=utf-8 python scripts/probe_men_races.py --days 2 --limit 40
  PYTHONIOENCODING=utf-8 python scripts/probe_men_races.py --date 2026-08-08 --limit 20
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_schedule import (
    build_kaisai_list_url, parse_kaisai_list, kaisai_race_date,
    fetch_race_numbers, fetch_girls_race_numbers,
)
from src.collect.dataset import collect_race_dataset
from db.repository import DatasetRepo

PROBE_DB = ROOT / "data" / "keirin_men_probe.sqlite"


def men_race_numbers(k) -> list[int]:
    """開催の男子レース番号＝全レース − ガールズレース。"""
    allr = set(fetch_race_numbers(k))
    girls = set(fetch_girls_race_numbers(k))
    return sorted(allr - girls)


def main() -> None:
    ap = argparse.ArgumentParser(description="男子競輪の試験収集（Phase 0・プローブDBへ）")
    ap.add_argument("--date", help="対象日 YYYY-MM-DD（既定=昨日から遡る）")
    ap.add_argument("--days", type=int, default=2, help="遡る日数")
    ap.add_argument("--limit", type=int, default=40, help="収集レース数の上限（必ず切る）")
    ap.add_argument("--interval", type=float, default=1.2, help="リクエスト間隔(秒)")
    ap.add_argument("--db", default=str(PROBE_DB))
    args = ap.parse_args()

    set_default_interval(args.interval)
    start = date.fromisoformat(args.date) if args.date else (date.today() - timedelta(days=1))
    days = [start - timedelta(days=i) for i in range(args.days)]

    repo = DatasetRepo(args.db)
    print(f"プローブDB: {args.db}（本番DBには書きません）")
    print(f"対象日: {[d.isoformat() for d in days]}  上限{args.limit}R  間隔{args.interval}s\n")

    got = 0
    miss = Counter()          # 取得できなかった項目
    fields = Counter()        # 車立ての分布
    fails: list[str] = []
    for d in days:
        if got >= args.limit:
            break
        try:
            res = fetch(build_kaisai_list_url(d.year, d.month, d.day))
            kais = [k for k in parse_kaisai_list(res.text)
                    if kaisai_race_date(k.kaisai_day_code) == d]
        except Exception as e:
            print(f"{d}: 開催一覧の取得失敗 {type(e).__name__}: {e}")
            continue
        print(f"--- {d}: 開催{len(kais)}件 ---")
        for k in kais:
            if got >= args.limit:
                break
            try:
                nums = men_race_numbers(k)
            except Exception as e:
                print(f"  {k.venue_code}: レース番号取得失敗 {type(e).__name__}: {e}")
                continue
            if not nums:
                continue
            print(f"  会場{k.venue_code} ガールズ開催={k.is_girls} 男子{len(nums)}R -> {nums[:12]}")
            for rno in nums:
                if got >= args.limit:
                    break
                try:
                    ds = collect_race_dataset(k, rno, require_girls=False)
                except Exception as e:
                    fails.append(f"{k.venue_code} R{rno}: {type(e).__name__}: {e}")
                    print(f"    R{rno} 収集失敗: {type(e).__name__}: {e}")
                    continue
                rid = ds.race_id
                rdate = kaisai_race_date(k.kaisai_day_code).isoformat()
                repo.save_race(rid, rdate, k.venue_code, rno, ds.is_girls,
                               ds.deadline, ds.field_size)
                if ds.entries:
                    repo.save_entries(rid, ds.entries)
                else:
                    miss["出走表"] += 1
                if getattr(ds, "recent", None):
                    repo.save_recent_form(rid, ds.recent)
                else:
                    miss["直近成績"] += 1
                nb = getattr(ds, "narabi", None)
                if nb and nb.get("order"):
                    repo.save_narabi(rid, nb)
                else:
                    miss["並び予想"] += 1
                if ds.odds_final:
                    repo.save_odds_final(rid, ds.odds_final)
                else:
                    miss["確定オッズ"] += 1
                if ds.results:
                    repo.save_results(rid, ds.results)
                else:
                    miss["着順"] += 1
                if ds.payout:
                    repo.save_payout(rid, ds.payout)
                else:
                    miss["払戻"] += 1
                fields[ds.field_size] += 1
                got += 1
                cls = {e.class_rank for e in ds.entries}
                print(f"    R{rno} OK 車立て{ds.field_size} 級班{sorted(cls)} "
                      f"オッズ{len(ds.odds_final)} 着順{len(ds.results)} "
                      f"並び{len((nb or {}).get('order') or [])}")

    print(f"\n=== 収集 {got}R ===")
    print("車立て分布:", dict(sorted(fields.items())))
    print("取得できなかった項目:", dict(miss) if miss else "なし")
    if fails:
        print(f"収集失敗 {len(fails)}件:")
        for f in fails[:10]:
            print("  " + f)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
