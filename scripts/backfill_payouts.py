"""三連複・二車単の払戻を過去レースへバックフィルする（結果ページ1フェッチ/レース）。

三連単と**同じ結果ページの払戻テーブル**に三連複(`a=b=c`)・二車単(末尾の `a-b`)が
入っている（2026-08 確認）。**1レース1リクエストで両方を保存**するため、
既に三連複だけ入っているレースでも二車単のためだけに再取得することはない
（trio/exacta のどちらかが欠けているレースだけを対象にする）。

なぜ要るか: 三連複・二車単を数点に絞る買い方の回収率を実測するため。的中率はモデル分布から
出せるが、回収率は実払戻が無いと測れない（別プールなので合成オッズと一致する保証がない）。

  PYTHONIOENCODING=utf-8 python scripts/backfill_payouts.py --db data/keirin.sqlite --days 1200
  python scripts/backfill_payouts.py --db data/keirin_men.sqlite --limit 50   # 試し
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
from src.collect.gamboo_result import (build_result_url, parse_trio_payout,
                                        parse_exacta_payout)
from src.collect.gamboo_schedule import (build_kaisai_list_url, parse_kaisai_list,
                                         fetch_race_numbers_for, kaisai_race_date)


def main() -> None:
    ap = argparse.ArgumentParser(description="三連複・二車単払戻のバックフィル")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--days", type=int, default=1200, help="今日から遡る日数")
    ap.add_argument("--limit", type=int, help="取得レース数の上限（試し実行用）")
    ap.add_argument("--interval", type=float, default=1.2, help="リクエスト間隔（秒）")
    args = ap.parse_args()

    set_default_interval(args.interval)      # 課題G: 1秒以上あける
    repo = DatasetRepo(args.db)              # payouts_exacta 等の新テーブルもここで作られる

    # trio か exacta のどちらかが欠けているレースだけを対象にする。
    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    have_trio = {r[0] for r in c.execute("SELECT race_id FROM payouts_trio")}
    have_exacta = {r[0] for r in c.execute("SELECT race_id FROM payouts_exacta")}
    want = defaultdict(list)
    for rid, d in c.execute(
            "SELECT r.race_id, r.race_date FROM races r"
            " JOIN payouts_trifecta p ON p.race_id=r.race_id"):
        if rid not in have_trio or rid not in have_exacta:
            want[d].append(rid)
    c.close()
    today = dt.date.today()
    dates = sorted((d for d in want if
                    (today - dt.date.fromisoformat(d)).days <= args.days), reverse=True)
    total = sum(len(want[d]) for d in dates)
    print(f"三連複 {len(have_trio):,} / 二車単 {len(have_exacta):,} 取得済 "
          f"・未完了 {total:,}レース（{len(dates)}日分）を再取得")
    if not total:
        return

    got = miss = 0
    t0 = time.time()
    for d in dates:
        target = dt.date.fromisoformat(d)
        want_d = set(want[d])
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
                if rid not in want_d:
                    continue
                try:
                    h = fetch(build_result_url(k.kaisai_code, k.kaisai_day_code, rno)).text
                    tri = parse_trio_payout(h)
                    exa = parse_exacta_payout(h)
                except Exception:
                    tri = exa = None
                saved = False
                if tri and rid not in have_trio:
                    repo.save_trio_payout(rid, tri); have_trio.add(rid); saved = True
                if exa and rid not in have_exacta:
                    repo.save_exacta_payout(rid, exa); have_exacta.add(rid); saved = True
                if saved:
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
