"""リアルタイム三連単オッズの軽量スナップショット取得（時系列蓄積・検証基盤）。

予測はせず、対象窓内（締切まで --within 分以内・過ぎていない）のガールズ各レースの三連単オッズだけを
取得し、取得時刻付きで data/odds_snapshots.sqlite に保存する。締切30分前からの1分更新
(refresh_predictions) より**広い窓（既定120分）を粗い間隔**で回すことで、「AIの自動購入で潰れる前の
ソフトなオッズ」も時系列で捉える。実現ROIは確定配当(payouts_trifecta)で決まる（パリミチュアル）ため、
本スナップショットは「その時刻に見えていたオッズで買うと判定したか」を後から再現する材料。

  python scripts/snapshot_odds.py --within 120        # 締切120分前〜締切のオッズを1回取得・保存
軽量化: data.json の締切をキャッシュし、窓外レースは取得自体をスキップ。オッズページ1フェッチ/レース。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bs4 import BeautifulSoup
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_schedule import (
    build_kaisai_list_url, parse_kaisai_list, fetch_girls_race_numbers, kaisai_race_date,
)
from src.collect.gamboo_odds import build_odds_url, parse_trifecta_odds, parse_deadline
from src.collect.snapshot import build_race_id
from db.repository import SnapshotRepo

SNAPSHOT_DB = ROOT / "data" / "odds_snapshots.sqlite"
DATA_JSON = ROOT / "dashboard" / "data.json"
JST = timezone(timedelta(hours=9))


def _venue_map(html: str) -> dict[str, str]:
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


def _mins_to_deadline(deadline: str, now: datetime) -> float | None:
    if not deadline or ":" not in deadline:
        return None
    h, m = (int(x) for x in deadline.split(":"))
    dl = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return (dl - now).total_seconds() / 60.0


def main() -> None:
    ap = argparse.ArgumentParser(description="リアルタイム三連単オッズの時系列取得")
    ap.add_argument("--within", type=float, default=120.0, help="締切まで何分以内を対象にするか")
    args = ap.parse_args()
    set_default_interval(0.5)
    now = datetime.now(JST)
    target = now.date()

    # data.json の締切キャッシュ（窓外レースの無駄フェッチ回避）
    dl_cache = {}
    if DATA_JSON.exists():
        try:
            doc = json.loads(DATA_JSON.read_text(encoding="utf-8"))
            dl_cache = {(r.get("venue"), r.get("race_no")): r.get("deadline")
                        for r in doc.get("predictions", {}).get("races", [])}
        except Exception:
            pass

    res = fetch(build_kaisai_list_url(target.year, target.month, target.day))
    kaisai_list = [k for k in parse_kaisai_list(res.text)
                   if k.is_girls and kaisai_race_date(k.kaisai_day_code) == target]
    venues = _venue_map(res.text)

    try:
        repo = SnapshotRepo(SNAPSHOT_DB)
    except Exception as e:
        print(f"snapshot DB open失敗: {e}"); return

    saved = 0
    for k in kaisai_list:
        venue = venues.get(k.kaisai_code, k.venue_code)
        for rno in fetch_girls_race_numbers(k):
            dl = dl_cache.get((venue, rno))
            if dl:                                      # 締切既知→窓外なら取得しない
                m = _mins_to_deadline(dl, now)
                if m is None or m < 0 or m > args.within:
                    continue
            try:
                html = fetch(build_odds_url(k.kaisai_code, k.kaisai_day_code, rno)).text
                odds = {kk: v for kk, v in parse_trifecta_odds(html).items() if v and v < 9999}
                if not dl:                              # 締切未知→取得後に窓判定
                    m = _mins_to_deadline(parse_deadline(html) or "", now)
                    if m is None or m < 0 or m > args.within:
                        continue
                if odds:
                    repo.save_snapshot(build_race_id(k.kaisai_day_code, rno), odds, now)
                    saved += 1
            except Exception as e:
                print(f"  {venue} R{rno} オッズ取得失敗: {e}")
                continue
    print(f"snapshot保存: {saved}レース @ {now:%H:%M} (within {args.within:.0f}分)")


if __name__ == "__main__":
    main()
