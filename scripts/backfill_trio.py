"""三連複の払戻を過去レースへバックフィルする（結果ページ1フェッチ/レース）。

三連単と**同じ結果ページの払戻テーブル**に `a=b=c` 形式で入っている（2026-08-18確認）。
オッズ全点より遥かに軽く、1レース1リクエストで済む。

**なぜ要るか**: 三連複を数点に絞る買い方の回収率を測るため。的中率はモデル分布から
一意に出せる（三連単分布の6順列を足すだけ）が、**回収率は三連複の払戻が無いと測れない**。
市場が賭式間で整合していれば三連複の配当は6通りの合成オッズと一致し回収率も同じになるが、
プールは別なので実測しないと分からない。

  PYTHONIOENCODING=utf-8 python scripts/backfill_trio.py --days 400
  python scripts/backfill_trio.py --days 30 --limit 50     # 試し
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR
from db.repository import DatasetRepo
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_result import build_result_url, parse_trio_payout
from src.collect.gamboo_schedule import (build_kaisai_list_url, parse_kaisai_list,
                                         fetch_race_numbers_for, kaisai_race_date)


def main() -> None:
    ap = argparse.ArgumentParser(description="三連複払戻のバックフィル")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--days", type=int, default=400, help="今日から遡る日数")
    ap.add_argument("--limit", type=int, help="取得レース数の上限（試し実行用）")
    ap.add_argument("--interval", type=float, default=1.2, help="リクエスト間隔（秒）")
    args = ap.parse_args()

    set_default_interval(args.interval)      # 課題G: 1秒以上あける
    repo = DatasetRepo(args.db)

    # 既に持っているレースは飛ばす。日付ごとに何レース残っているかを先に数える
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    have = {r[0] for r in c.execute("SELECT race_id FROM payouts_trio")}
    want = defaultdict(list)
    for rid, d in c.execute(
            "SELECT r.race_id, r.race_date FROM races r"
            " JOIN payouts_trifecta p ON p.race_id=r.race_id"):
        if rid not in have:
            want[d].append(rid)
    c.close()
    today = dt.date.today()
    dates = sorted((d for d in want if
                    (today - dt.date.fromisoformat(d)).days <= args.days), reverse=True)
    total = sum(len(want[d]) for d in dates)
    print(f"三連複を持っているレース {len(have):,} / 未取得 {total:,}（{len(dates)}日分）")
    if not total:
        return

    got = miss = 0
    t0 = time.time()
    for d in dates:
        target = dt.date.fromisoformat(d)
        try:
            ks = parse_kaisai_list(fetch(build_kaisai_list_url(
                target.year, target.month, target.day)).text)
        except Exception as e:
            print(f"  {d}: 開催一覧の取得に失敗 {e}")
            continue
        ks = [k for k in ks if kaisai_race_date(k.kaisai_day_code) == target]
        for k in ks:
            try:
                nos = fetch_race_numbers_for(k, "all")
            except Exception:
                continue
            for rno in nos:
                rid = f"{k.kaisai_day_code}{rno:02d}"
                if rid in have or rid not in set(want[d]):
                    continue
                try:
                    h = fetch(build_result_url(k.kaisai_code, k.kaisai_day_code, rno)).text
                    p = parse_trio_payout(h)
                except Exception:
                    p = None
                if p:
                    repo.save_trio_payout(rid, p)
                    have.add(rid)
                    got += 1
                else:
                    miss += 1
                if args.limit and got >= args.limit:
                    print(f"\n上限 {args.limit} に到達")
                    print(f"取得 {got:,} / 取れず {miss:,} / {time.time()-t0:.0f}秒")
                    return
        el = time.time() - t0
        rate = got / el * 60 if el else 0
        print(f"  {d}: 累計 取得{got:,} 取れず{miss:,}  "
              f"({rate:.0f}件/分, 残り約{(total-got-miss)/max(rate,1):.0f}分)")

    print(f"\n完了: 取得 {got:,} / 取れず {miss:,} / {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
