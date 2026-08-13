"""【Phase 1】男子競輪の本収集（直近1年）。**再開可能**な長時間バッチ。

規模: 1日あたり男子は約90レース × 365日 ≒ 33,000レース。1レース2フェッチ＋間隔1.2秒で
**27時間規模**になるため、途中終了しても続きから再開できることを最優先に設計している。

  - 収集済みレース（races に race_id がある）は**フェッチせずスキップ**
  - 開催一覧・レース番号一覧も日単位でキャッシュ（同じ日を2度引かない）
  - Ctrl+C / プロセス終了で中断しても、同じコマンドを再実行すれば続きから
  - 保存先は男子専用DB `data/keirin_men.sqlite`（ガールズ本番DBとは分離）

分離しておく理由: 男女は同一レースを走らないので Elo も選手プールも交わらない。
将来まとめたくなればスキーマが同じなので後から統合できる（逆は面倒）。

  PYTHONIOENCODING=utf-8 python scripts/collect_men.py --days 365
  PYTHONIOENCODING=utf-8 python scripts/collect_men.py --days 365 --limit 2000   # 分割実行
  PYTHONIOENCODING=utf-8 python scripts/collect_men.py --stats                   # 進捗だけ見る
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import traceback
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_schedule import (
    build_kaisai_list_url, parse_kaisai_list, kaisai_race_date,
    fetch_race_numbers, fetch_girls_race_numbers,
)
from src.collect.dataset import collect_race_dataset
from src.collect.snapshot import build_race_id
from db.repository import DatasetRepo

MEN_DB = ROOT / "data" / "keirin_men.sqlite"


def _done_ids(db: str) -> set[str]:
    """収集済み race_id。再開時にフェッチを丸ごと省くための土台。"""
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        ids = {r[0] for r in c.execute("SELECT race_id FROM races")}
        c.close()
        return ids
    except Exception:
        return set()


def _stats(db: str) -> None:
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except Exception:
        print("DBがまだありません")
        return
    n = c.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    if not n:
        print("収集0件")
        c.close()
        return
    lo, hi = c.execute("SELECT MIN(race_date),MAX(race_date) FROM races").fetchone()
    fs = c.execute("SELECT field_size,COUNT(*) FROM races GROUP BY field_size ORDER BY field_size").fetchall()
    res = c.execute("SELECT COUNT(DISTINCT race_id) FROM results").fetchone()[0]
    nb = c.execute("SELECT COUNT(DISTINCT race_id) FROM narabi").fetchone()[0]
    ln = c.execute("SELECT COUNT(DISTINCT race_id) FROM narabi WHERE line_id IS NOT NULL").fetchone()[0]
    days = c.execute("SELECT COUNT(DISTINCT race_date) FROM races").fetchone()[0]
    c.close()
    print(f"収集 {n:,}レース / {days}日分 / 期間 {lo}〜{hi}")
    print(f"  車立て: {dict(fs)}")
    print(f"  結果あり {res:,}  並び予想あり {nb:,}  うちライン情報あり {ln:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="男子競輪の本収集（再開可能）")
    ap.add_argument("--days", type=int, default=365, help="今日から遡る日数")
    ap.add_argument("--end", help="収集の起点日 YYYY-MM-DD（既定=昨日）")
    ap.add_argument("--limit", type=int, help="今回の実行で収集する上限レース数（分割実行用）")
    ap.add_argument("--interval", type=float, default=1.2, help="リクエスト間隔(秒)")
    ap.add_argument("--db", default=str(MEN_DB))
    ap.add_argument("--stats", action="store_true", help="進捗だけ表示して終了")
    args = ap.parse_args()

    if args.stats:
        _stats(args.db)
        return

    set_default_interval(args.interval)
    end = date.fromisoformat(args.end) if args.end else (date.today() - timedelta(days=1))
    days = [end - timedelta(days=i) for i in range(args.days)]
    done = _done_ids(args.db)
    repo = DatasetRepo(args.db)

    print(f"男子DB: {args.db}（ガールズ本番DBとは分離）")
    print(f"期間: {days[-1]} 〜 {days[0]}（{len(days)}日）  既収集 {len(done):,}R"
          + (f"  今回上限 {args.limit:,}R" if args.limit else "  上限なし"))
    print("中断しても同じコマンドで再開できます\n")

    t0 = time.time()
    got = skip = 0
    fails = Counter()
    for d in days:
        if args.limit and got >= args.limit:
            break
        try:
            res = fetch(build_kaisai_list_url(d.year, d.month, d.day))
            kais = [k for k in parse_kaisai_list(res.text)
                    if kaisai_race_date(k.kaisai_day_code) == d]
        except Exception as e:
            fails[f"開催一覧:{type(e).__name__}"] += 1
            continue
        for k in kais:
            if args.limit and got >= args.limit:
                break
            # レース番号一覧を引く前に、その開催が丸ごと収集済みかを判定できないので
            # ここは引く。ただし1開催あたり2フェッチなので全体からすれば軽い。
            try:
                nums = sorted(set(fetch_race_numbers(k)) - set(fetch_girls_race_numbers(k)))
            except Exception as e:
                fails[f"レース番号:{type(e).__name__}"] += 1
                continue
            for rno in nums:
                if args.limit and got >= args.limit:
                    break
                rid = build_race_id(k.kaisai_day_code, rno)
                if rid in done:                 # ★再開の要。フェッチせず飛ばす
                    skip += 1
                    continue
                try:
                    ds = collect_race_dataset(k, rno, require_girls=False)
                except Exception as e:
                    fails[f"収集:{type(e).__name__}"] += 1
                    continue
                repo.save_race(ds.race_id, d.isoformat(), k.venue_code, rno,
                               ds.is_girls, ds.deadline, ds.field_size,
                               grade=ds.grade, race_name=ds.race_name)
                if ds.entries:
                    repo.save_entries(ds.race_id, ds.entries)
                if getattr(ds, "recent", None):
                    repo.save_recent_form(ds.race_id, ds.recent)
                nb = getattr(ds, "narabi", None)
                if nb and nb.get("order"):
                    repo.save_narabi(ds.race_id, nb)
                if ds.odds_final:
                    repo.save_odds_final(ds.race_id, ds.odds_final)
                if ds.results:
                    repo.save_results(ds.race_id, ds.results)
                if ds.payout:
                    repo.save_payout(ds.race_id, ds.payout)
                done.add(rid)
                got += 1
                if got % 50 == 0:
                    el = time.time() - t0
                    rate = got / el
                    print(f"  [{datetime.now():%m-%d %H:%M}] {got:,}R収集 "
                          f"(スキップ{skip:,}) {rate*3600:,.0f}R/時 "
                          f"経過{el/3600:.1f}h  最新={d} {k.venue_code} R{rno}")

    el = time.time() - t0
    print(f"\n=== 今回 {got:,}R 収集 / {skip:,}R スキップ（既収集）/ {el/3600:.2f}時間 ===")
    if fails:
        print("失敗:", dict(fails))
    _stats(args.db)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました。同じコマンドで再開できます。")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
