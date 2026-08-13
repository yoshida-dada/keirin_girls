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
    build_kaisai_list_url, parse_kaisai_list, fetch_race_numbers_for, kaisai_race_date,
)
from src.collect.dataset import collect_race_dataset
from src.collect.snapshot import build_race_id
from db.repository import DatasetRepo, combo_to_str

DATA_JSON = ROOT / "dashboard" / "data.json"
DATA_JSON_MEN = ROOT / "dashboard" / "data_men.json"   # 男子は別ページ・別データ
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


def fetch_and_store(target: date, db_path: Path, min_after: int = 20,
                    include: str = "girls", men_db: Path | None = None) -> int:
    """締切+min_after分を過ぎた未取得結果を取得・DB格納・data.json反映。件数を返す。

    include="men"/"all" のとき男子は **men_db（既定 data/keirin_men.sqlite）** へ入れる。
    予測時の選手成績・展開特徴・Elo は男子DBから引くので、日々の結果が別DBに落ちると
    男子の履歴だけ止まる。学習母集団を混ぜないためにもDBは分けたまま書き分ける。
    """
    now = datetime.now(JST)
    set_default_interval(0.6)
    res = fetch(build_kaisai_list_url(target.year, target.month, target.day))
    kaisai_list = [k for k in parse_kaisai_list(res.text)
                   if kaisai_race_date(k.kaisai_day_code) == target]
    if include == "girls":
        kaisai_list = [k for k in kaisai_list if k.is_girls]
    venues = _venue_map(res.text)

    # 男女でデータファイルが別なので両方開き、レースを跨いだ辞書で引く。
    # 片方だけ見ると、もう一方の結果が永久に反映されない。
    docs = []
    for path in (DATA_JSON, DATA_JSON_MEN):
        d = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        docs.append((path, d))
    races = [r for _, d in docs for r in ((d.get("predictions") or {}).get("races") or [])]
    # data.json は前後1日を持つため (date, venue, race_no) で引く。日付を外すと
    # 同一会場のR番号が日をまたいで衝突し、別日のレースへ結果を書き込んでしまう。
    tstr = target.isoformat()
    by_key = {(r.get("date") or tstr, r.get("venue"), r.get("race_no")): r for r in races}

    men_db = men_db or (Path(db_path).parent / "keirin_men.sqlite")
    repo = DatasetRepo(str(db_path))
    repo_men = DatasetRepo(str(men_db)) if include != "girls" else None
    updated = 0
    try:
        for k in kaisai_list:
            venue = venues.get(k.kaisai_code, k.venue_code)
            for rno in fetch_race_numbers_for(k, include):
                drace = by_key.get((tstr, venue, rno))
                if drace and drace.get("result"):
                    continue                                  # 取得済み
                deadline = drace.get("deadline") if drace else None
                if min_after > 0 and deadline and ":" in str(deadline):
                    h, m = (int(x) for x in str(deadline).split(":"))
                    dl = now.replace(hour=h, minute=m, second=0, microsecond=0)
                    dl = dl.replace(year=target.year, month=target.month, day=target.day)
                    if (now - dl).total_seconds() < min_after * 60:
                        continue                              # 締切+min_after分に未達
                try:
                    # 完全収集: オッズページ(出走表/直近成績/確定オッズ)＋結果ページ(着順/払戻)を一括取得
                    ds = collect_race_dataset(k, rno, require_girls=(include == "girls"))
                except Exception as e:
                    print(f"  {venue} R{rno} 収集失敗: {e}")
                    continue
                if not ds.results:
                    continue                                  # 未確定
                race_id = ds.race_id
                race_date = kaisai_race_date(k.kaisai_day_code).isoformat()
                # 男子はガールズDBに混ぜない（学習母集団が別・予測時も別DBを引く）
                sink = repo if ds.is_girls or repo_men is None else repo_men
                # 出走表/直近成績/確定オッズ/着順/払戻 を保存（リアルタイムvs最終の検証土台）
                sink.save_race(race_id, race_date, k.venue_code, rno,
                               ds.is_girls, ds.deadline, ds.field_size,
                               grade=ds.grade, race_name=ds.race_name)
                if ds.entries:
                    sink.save_entries(race_id, ds.entries)
                if ds.recent:
                    sink.save_recent_form(race_id, ds.recent)
                if ds.odds_final:
                    sink.save_odds_final(race_id, ds.odds_final)
                if ds.narabi and ds.narabi.get("order"):
                    sink.save_narabi(race_id, ds.narabi)
                sink.save_results(race_id, ds.results)
                sink.save_payout(race_id, ds.payout)
                if drace is not None:
                    drace["result"] = _result_section(ds.results, ds.payout, drace)
                    updated += 1
                print(f"  {venue} R{rno} 完全収集: 1着{ds.results[0].car_number}車"
                      f"（出走{len(ds.entries)}/直近{len(ds.recent)}/確定オッズ{len(ds.odds_final)}）")
    finally:
        repo.close()
        if repo_men is not None:
            repo_men.close()

    if updated and races:
        # drace はどちらかの doc 内のオブジェクトを直接書き換えているので、両方保存する
        for path, d in docs:
            if not d.get("predictions"):
                continue
            d["predictions"]["results_updated"] = now.strftime("%Y-%m-%d %H:%M JST")
            path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    return updated


def main() -> None:
    ap = argparse.ArgumentParser(description="確定結果の取得・DB格納・data.json反映")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--date", help="対象日 YYYY-MM-DD（既定=今日）")
    ap.add_argument("--min-after", type=int, default=20, help="締切から何分後以降を取得対象にするか")
    ap.add_argument("--window", type=int, default=0,
                    help="対象日の前後何日も取得するか（既定=0。1で昨日ぶんの取りこぼしも回収）")
    ap.add_argument("--include", choices=["girls", "men", "all"], default="girls",
                    help="取得対象（既定=girls）。男子は data/keirin_men.sqlite へ格納する")
    args = ap.parse_args()
    target = date.fromisoformat(args.date) if args.date else datetime.now(JST).date()
    total = 0
    for i in range(-args.window, 1):          # 過去側のみ（未来のレースに結果は無い）
        d = target + timedelta(days=i)
        total += fetch_and_store(d, Path(args.db), min_after=args.min_after,
                                 include=args.include)
    print(f"結果反映: {total}レース")


if __name__ == "__main__":
    main()
