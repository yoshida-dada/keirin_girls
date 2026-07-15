"""確定レース結果の取得・DB格納・data.json反映（B: 更新周期／結果取得）。

締切から約20分以上経過したガールズ各レースについて、GambooBETの結果ページから
着順(results)・三連単払戻(payouts)を取得してDBへ格納し、dashboard/data.json の該当レースに
`result` セクション（着順・上り・決まり手・払戻・予測的中）を付与する。既に結果があるレースや
未確定レースはスキップ。DB格納した当日結果は翌朝のbuildで meet_results/style_counts/Elo/学習に
自動的に反映される（＝翌日以降の予測に活用）。

  python scripts/fetch_results.py                 # 本日分（締切+20分経過レース）
  python scripts/fetch_results.py --date 2026-07-14 --min-after 0   # 過去日の全確定レース

結果ページ1フェッチ/レース（結果行の rider_name を含むため、履歴系特徴は entries 無しでも成立）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_schedule import (
    build_kaisai_list_url, parse_kaisai_list, fetch_girls_race_numbers, kaisai_race_date,
)
from src.collect.gamboo_result import fetch_result
from src.collect.snapshot import build_race_id
from db.repository import DatasetRepo, combo_to_str

DATA_JSON = ROOT / "dashboard" / "data.json"
DEFAULT_DB = ROOT / "data" / "keirin.sqlite"
JST = timezone(timedelta(hours=9))


def _venue_map(html: str) -> dict[str, str]:
    """開催一覧HTML → {開催コード: 会場名}（build_predictions と同じ規則）。"""
    soup = BeautifulSoup(html, "html.parser")
    ul = soup.find("ul", class_="kaisai_list")
    out: dict[str, str] = {}
    if not ul:
        return out
    for li in ul.find_all("li", recursive=False):
        a = li.find("a", href=re.compile(r"/race-list/"))
        if not a:
            continue
        parts = a["href"].split("/race-list/")[1].strip("/").split("/")
        name = li.get_text(" ", strip=True)
        m = re.search(r"([^\s]+?競輪)", name)
        out[parts[0]] = m.group(1) if m else name[:8]
    return out


def _result_section(rows: list, payout, drace: dict | None) -> dict:
    """結果行＋払戻から data.json 用の result セクションを組む（予測的中も算出）。"""
    order = [{"pos": r.position, "car": r.car_number, "name": r.rider_name,
              "last_lap": r.last_lap, "kimarite": r.kimarite, "sb": r.sb}
             for r in rows]
    top3 = [r.car_number for r in rows if r.position in (1, 2, 3)]
    actual_tri = tuple(top3[:3]) if len(top3) >= 3 else None

    hit = {}
    if drace:
        riders = drace.get("riders") or []
        if riders and order:
            first = next((o["car"] for o in order if o["pos"] == 1), None)
            hit["win_car"] = riders[0].get("car")           # モデル本命(1着確率トップ)
            hit["win_hit"] = (first is not None and first == riders[0].get("car"))
        # 実際の三連単がモデル確率で何位か（combos: [a,b,c,odds,prob,ev]）
        combos = drace.get("combos") or []
        if actual_tri and combos:
            ranked = sorted((c for c in combos if c[4] is not None), key=lambda c: -c[4])
            for i, c in enumerate(ranked, 1):
                if (c[0], c[1], c[2]) == actual_tri:
                    hit["tri_rank"] = i
                    break
            else:
                hit["tri_rank"] = None

    return {
        "order": order,
        "payout": ({"combo": "-".join(map(str, payout.combo)),
                    "yen": payout.payout, "pop": payout.popularity} if payout else None),
        "hit": hit,
        "fetched_at": datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
    }


def fetch_and_store(target: date, db_path: Path, min_after: int = 20) -> int:
    """締切+min_after分を過ぎた未取得ガールズ結果を取得・DB格納・data.json反映。件数を返す。"""
    now = datetime.now(JST)
    set_default_interval(0.6)
    res = fetch(build_kaisai_list_url(target.year, target.month, target.day))
    kaisai_list = [k for k in parse_kaisai_list(res.text)
                   if k.is_girls and kaisai_race_date(k.kaisai_day_code) == target]
    venues = _venue_map(res.text)

    doc = json.loads(DATA_JSON.read_text(encoding="utf-8")) if DATA_JSON.exists() else {}
    races = (doc.get("predictions") or {}).get("races") or []
    by_key = {(r.get("venue"), r.get("race_no")): r for r in races}

    repo = DatasetRepo(str(db_path))
    updated = 0
    try:
        for k in kaisai_list:
            venue = venues.get(k.kaisai_code, k.venue_code)
            for rno in fetch_girls_race_numbers(k):
                drace = by_key.get((venue, rno))
                if drace and drace.get("result"):
                    continue                                  # 取得済み
                deadline = drace.get("deadline") if drace else None
                if min_after > 0 and deadline and ":" in str(deadline):
                    h, m = (int(x) for x in str(deadline).split(":"))
                    dl = now.replace(hour=h, minute=m, second=0, microsecond=0)
                    if (now - dl).total_seconds() < min_after * 60:
                        continue                              # 締切+min_after分に未達
                try:
                    rows, payout = fetch_result(k.kaisai_code, k.kaisai_day_code, rno)
                except Exception as e:
                    print(f"  {venue} R{rno} 結果取得失敗: {e}")
                    continue
                if not rows:
                    continue                                  # 未確定
                race_id = build_race_id(k.kaisai_day_code, rno)
                race_date = kaisai_race_date(k.kaisai_day_code).isoformat()
                repo.save_race(race_id, race_date, k.venue_code, rno, True, deadline, len(rows))
                repo.save_results(race_id, rows)
                repo.save_payout(race_id, payout)
                if drace is not None:
                    drace["result"] = _result_section(rows, payout, drace)
                    updated += 1
                print(f"  {venue} R{rno} 結果格納: 1着{rows[0].car_number}車")
    finally:
        repo.close()

    if updated and races:
        doc["predictions"]["results_updated"] = now.strftime("%Y-%m-%d %H:%M JST")
        DATA_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated


def main() -> None:
    ap = argparse.ArgumentParser(description="確定結果の取得・DB格納・data.json反映")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--date", help="対象日 YYYY-MM-DD（既定=今日）")
    ap.add_argument("--min-after", type=int, default=20, help="締切から何分後以降を取得対象にするか")
    args = ap.parse_args()
    target = date.fromisoformat(args.date) if args.date else datetime.now(JST).date()
    n = fetch_and_store(target, Path(args.db), min_after=args.min_after)
    print(f"結果反映: {n}レース")


if __name__ == "__main__":
    main()
