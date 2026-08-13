"""既存レースに開催格(grade)とレース名(race_name)を後付けする（B-1）。

**開催一覧ページ1枚で開催の全日程ぶんが取れる**ので極めて安い。1ページに:
  - 開催ごと: `span.icon_grade grN`（格）/ `span.race`（開催名）
  - 開催の各日: `<table id="tbl_day{会場コード}{YYYYMMDD}">` に
      ヘッダ行 = 1R..12R、`td.name[colspan]` = レース名（colspanを展開してR番号へ割付）
      class の bg_s / bg_a / bg_l で S級 / A級 / ガールズ の別も付く
1開催が3〜7日ぶんの表を持つため、日付を1つ取ると前後の日も同時に埋まる。
既に埋まっている (会場,日付) はフェッチ自体を飛ばす＝再実行しても無駄打ちしない。

  python scripts/backfill_grade.py --dry-run          # 対象件数だけ見る
  python scripts/backfill_grade.py                    # ガールズ+男子DBを更新

**注意（精度の限界）**: このページの格は `icon_grade grN` のクラスだけで、全角表記
（"松山Ｇ１"）が無い。特殊開催（全プロ記念競輪ｉｎ武雄）で gr1 が実態とズレる例を確認済み。
1レース単位で正確に取るならオッズページの `parse_race_meta`（h1の全角表記が第一情報源）を
使うが、そちらは1レース1フェッチになる。バケット分析のG1/G2/G3とF1/F2の切り分けには
本方式で足りると判断した。
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

from config.settings import DATA_DIR
from src.collect.base import fetch, set_default_interval
from src.collect.gamboo_schedule import build_kaisai_list_url

# 実測の対応（men_keirin_plan.md 4.12.4）。未知の grN は警告して None にする。
# gr6=GP は 2025-12-28〜30 の平塚（競輪グランプリ）で実測して判明した。
# 調査時点では未確認だったが、未知クラスを警告する実装にしていたので拾えた。
GRADE_BY_CLASS = {"gr1": "F2", "gr2": "F1", "gr3": "G3", "gr4": "G2",
                  "gr5": "G1", "gr6": "GP"}
_TBL_ID = re.compile(r"^tbl_day(\d{2})(\d{8})$")


def parse_kaisai_page(html: str) -> dict[tuple[str, str], dict]:
    """開催一覧HTML → {(会場コード, YYYY-MM-DD): {"grade":..., "meet":..., "names":{R番号:名}}}。"""
    soup = BeautifulSoup(html, "html.parser")
    ul = soup.find("ul", class_="kaisai_list")
    out: dict[tuple[str, str], dict] = {}
    if not ul:
        return out
    unknown = set()
    for li in ul.find_all("li", recursive=False):
        h3 = li.find("h3", class_="title")
        grade = meet = None
        if h3:
            ic = h3.find("span", class_="icon_grade")
            for c in (ic.get("class") if ic else []) or []:
                if c.startswith("gr"):
                    grade = GRADE_BY_CLASS.get(c)
                    if grade is None:
                        unknown.add(c)
            rc = h3.find("span", class_="race")
            meet = rc.get_text(strip=True) if rc else None
        for tbl in li.find_all("table", id=_TBL_ID):
            m = _TBL_ID.match(tbl.get("id"))
            venue, ymd = m.group(1), m.group(2)
            d = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
            rows = tbl.find_all("tr")
            if not rows:
                continue
            # ヘッダ行の "1R","2R",… からレース番号の並びを取る
            nos = []
            for th in rows[0].find_all(["th", "td"]):
                mm = re.match(r"(\d+)R", th.get_text(strip=True))
                nos.append(int(mm.group(1)) if mm else None)
            names: dict[int, str] = {}
            for tr in rows[1:]:
                tds = tr.find_all("td", class_="name")
                if not tds:
                    continue
                i = 0
                for td in tds:                      # colspan を展開してR番号へ割り付ける
                    span = int(td.get("colspan") or 1)
                    nm = td.get_text(strip=True)
                    for j in range(i, min(i + span, len(nos))):
                        if nos[j]:
                            names[nos[j]] = nm
                    i += span
                break                                # レース名の行は1本だけ
            out[(venue, d)] = {"grade": grade, "meet": meet, "names": names}
    if unknown:
        print(f"  ⚠ 未知のグレードclass: {sorted(unknown)} → None のまま（要確認）")
    return out


def _targets(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """grade または race_name が欠けている (日付 → 会場コード集合)。"""
    out: dict[str, set[str]] = defaultdict(set)
    for d, v in conn.execute(
            "SELECT DISTINCT race_date, venue_code FROM races"
            " WHERE race_date IS NOT NULL AND (grade IS NULL OR race_name IS NULL)"):
        out[d].add(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="既存レースへ開催格・レース名を後付け")
    ap.add_argument("--dbs", nargs="*", default=["keirin.sqlite", "keirin_men.sqlite"])
    ap.add_argument("--interval", type=float, default=1.2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-days", type=int, help="今回の実行で処理する日数の上限（分割実行用）")
    args = ap.parse_args()
    set_default_interval(args.interval)

    conns = {}
    for name in args.dbs:
        p = DATA_DIR / name
        if p.exists():
            conns[name] = sqlite3.connect(str(p))
    if not conns:
        raise SystemExit("対象DBが無い")

    need: dict[str, set[str]] = defaultdict(set)
    for c in conns.values():
        for d, vs in _targets(c).items():
            need[d] |= vs
    total_pairs = sum(len(v) for v in need.values())
    print(f"未設定: {len(need)}日 / {total_pairs} (会場,日) ペア")
    if args.dry_run:
        ds = sorted(need)
        print(f"  範囲: {ds[0]} 〜 {ds[-1]}" if ds else "  なし")
        return

    done_pairs: set[tuple[str, str]] = set()
    updated = fetched = 0
    days = sorted(need)
    if args.limit_days:
        days = days[:args.limit_days]
    for i, d in enumerate(days, 1):
        # この日の会場が既に他の日のページで埋まっていれば取りに行かない
        if all((v, d) in done_pairs for v in need[d]):
            continue
        y, mo, dy = (int(x) for x in d.split("-"))
        try:
            html = fetch(build_kaisai_list_url(y, mo, dy)).text
        except Exception as e:
            print(f"  {d} 取得失敗: {e}")
            continue
        fetched += 1
        page = parse_kaisai_page(html)
        for (venue, pd_), info in page.items():
            done_pairs.add((venue, pd_))
            g, names = info["grade"], info["names"]
            if not g and not names:
                continue
            for c in conns.values():
                for rno, nm in (names or {}).items():
                    updated += c.execute(
                        "UPDATE races SET grade=COALESCE(?,grade), race_name=COALESCE(?,race_name)"
                        " WHERE race_date=? AND venue_code=? AND race_no=?",
                        (g, nm, pd_, venue, rno)).rowcount
                if not names and g:      # レース名表が無くても格だけは入れる
                    updated += c.execute(
                        "UPDATE races SET grade=COALESCE(?,grade)"
                        " WHERE race_date=? AND venue_code=?", (g, pd_, venue)).rowcount
        for c in conns.values():
            c.commit()
        if i % 20 == 0 or i == len(days):
            print(f"  [{i}/{len(days)}] {d}  取得{fetched}ページ / 更新{updated}行")

    print(f"\n完了: {fetched}ページ取得 / {updated}行更新")
    for name, c in conns.items():
        n = c.execute("SELECT COUNT(*) FROM races").fetchone()[0]
        g = c.execute("SELECT COUNT(*) FROM races WHERE grade IS NOT NULL").fetchone()[0]
        r = c.execute("SELECT COUNT(*) FROM races WHERE race_name IS NOT NULL").fetchone()[0]
        print(f"  {name}: grade {g}/{n} ({g/n:.1%}) / race_name {r}/{n} ({r/n:.1%})")
        c.close()


if __name__ == "__main__":
    main()
