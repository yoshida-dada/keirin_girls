"""競り（カッコ）の構造を過去レースへ後付けする。

**なぜ再フェッチが要るか**: DBには leg='競り' しか残っておらず、カッコの区切りが無い。
「連続する競り＝1グループ」で復元できるか実ページ8件で検証したところ、
**8件すべてが2グループ**だった（例: 9-(1・6)-(4・2)）。4人連続の競りは
392レース中92件（23.5%）あり、この復元は使えない。

対象は **leg='競り' を含むレースだけ**（392/25,383＝1.54%）。全件は取り直さない。

保存: narabi.seri_group（同じ位置を争うグループの通し番号。競りでない選手は NULL）
      pos_in_line は**触らない**。ここを変えると ln_mate/ln_third の意味が変わり、
      学習済みモデルとの train/inference skew になる。位置の解釈変更は再学習とセットで行う。

  PYTHONIOENCODING=utf-8 python scripts/backfill_seri.py --dry-run
  PYTHONIOENCODING=utf-8 python scripts/backfill_seri.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_odds import build_odds_url
from src.collect.gamboo_racecard import parse_narabi


def targets(conn: sqlite3.Connection) -> list[str]:
    """競りを含み、まだ seri_group が入っていないレース。"""
    have = {r[1] for r in conn.execute("PRAGMA table_info(narabi)")}
    if "seri_group" not in have:
        conn.execute("ALTER TABLE narabi ADD COLUMN seri_group INTEGER")
        conn.commit()
    rows = conn.execute(
        "SELECT race_id FROM narabi WHERE leg='競り'"
        " GROUP BY race_id HAVING SUM(CASE WHEN seri_group IS NOT NULL THEN 1 ELSE 0 END)=0")
    return [r[0] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="競り構造のバックフィル")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--interval", type=float, default=1.2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    set_default_interval(args.interval)

    conn = sqlite3.connect(args.db)
    ids = targets(conn)
    if args.limit:
        ids = ids[:args.limit]
    print(f"対象 {len(ids):,}レース（競りを含み未処理）")
    if args.dry_run:
        conn.close()
        return

    done = fail = groups = 0
    for i, rid in enumerate(ids, 1):
        day_code, kaisai_code, rno = rid[:14], rid[:10], int(rid[14:])
        try:
            html = fetch(build_odds_url(kaisai_code, day_code, rno)).text
            nb = parse_narabi(html)
        except Exception as e:
            print(f"  {rid} 取得失敗: {e}")
            fail += 1
            continue
        if not nb["order"]:
            fail += 1
            continue
        # 競りグループに通し番号を振る。グループに属さない選手は NULL のまま
        for gi, g in enumerate(nb["seri"]):
            groups += 1
            for car in g:
                conn.execute(
                    "UPDATE narabi SET seri_group=? WHERE race_id=? AND car_number=?",
                    (gi, rid, car))
        done += 1
        if i % 25 == 0 or i == len(ids):
            conn.commit()
            print(f"  [{i}/{len(ids)}] 更新{done} / 失敗{fail} / グループ{groups}")
    conn.commit()

    n = conn.execute("SELECT COUNT(DISTINCT race_id) FROM narabi"
                     " WHERE seri_group IS NOT NULL").fetchone()[0]
    m = conn.execute("SELECT COUNT(*) FROM narabi WHERE seri_group IS NOT NULL").fetchone()[0]
    print(f"\n完了: {done}レース更新 / 失敗{fail}")
    print(f"  seri_group あり: {n:,}レース / {m:,}選手")
    conn.close()


if __name__ == "__main__":
    main()
