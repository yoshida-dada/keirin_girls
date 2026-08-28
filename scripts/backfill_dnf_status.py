"""過去のDNF在籍レースを再スクレイプし、着ステータス(落/失/欠)を dnf_status に backfill。

現状DBは落車/失格/欠が全て position NULL で区別できない。結果ページの着列(落/失/欠)を取得して
dnf_status(race_id,car_number,rider_name,status) に格納。落車明けバッジ・特徴の基盤。
既に dnf_status にある race_id はスキップ（再開可能）。1秒間隔（規約）。

  PYTHONIOENCODING=utf-8 python scripts/backfill_dnf_status.py --db data/keirin_men.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_result import build_result_url, parse_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    args = ap.parse_args()
    set_default_interval(1.0)

    ro = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    ro.execute("PRAGMA query_only=1")
    dnf_rids = [r[0] for r in ro.execute(
        "SELECT DISTINCT race_id FROM results WHERE position IS NULL ORDER BY race_id")]
    # 既に処理済み（dnf_statusに在る）race_id
    done = set()
    try:
        done = {r[0] for r in ro.execute("SELECT DISTINCT race_id FROM dnf_status")}
    except sqlite3.OperationalError:
        pass
    ro.close()
    todo = [r for r in dnf_rids if r not in done]
    print(f"DNF在籍レース {len(dnf_rids):,} / 済 {len(done):,} / 今回 {len(todo):,} を再取得（1秒間隔）")

    conn = sqlite3.connect(args.db)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS dnf_status ("
        " race_id TEXT NOT NULL, car_number INTEGER NOT NULL, rider_name TEXT,"
        " status TEXT, PRIMARY KEY (race_id, car_number))")
    conn.commit()

    ok = fail = saved = 0
    t0 = time.time()
    for i, rid in enumerate(todo, 1):
        kc, dc, rno = rid[:10], rid[:14], int(rid[14:])
        try:
            html = fetch(build_result_url(kc, dc, rno)).text
            rows = parse_results(html)
            recs = [(rid, r.car_number, r.rider_name, r.status)
                    for r in rows if (r.status or "")]
            if recs:
                conn.executemany("INSERT OR REPLACE INTO dnf_status VALUES (?,?,?,?)", recs)
                conn.commit()
                saved += len(recs)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 10:
                print(f"  {rid} 失敗: {type(e).__name__}: {e}")
        if i % 100 == 0:
            rate = i / (time.time() - t0) * 60
            print(f"  {i}/{len(todo)}  取得{ok} 失敗{fail} status保存{saved}  ({rate:.0f}件/分)")
    conn.close()
    print(f"完了: 取得{ok} 失敗{fail} status{saved}行  {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
