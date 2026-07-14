"""GambooBET（楽天Kドリーム）三連単オッズのスクレイパー（S1、主系）。

ページ構造（2026-07調査）:
  URL: https://keirin.kdreams.jp/gamboo/keirin-kaisai/race-card/odds/{開催}/{開催日}/{R}/3rentan/
  1着車番ごとに <table class="odds_table bt5 {1着車番}"> が1枚。各テーブル内:
    - 列ヘッダ行 = 3着車番
    - 各データ行の先頭 <th> = 2着車番、続く <td> = そのオッズ（class="empty" は不可能な組合せ）
  → combo (1着,2着,3着) → オッズ を全 N*(N-1)*(N-2) 点抽出する。

7車立て(ガールズ)は210点。オッズは締切前は暫定・締切後は確定。取得時刻は呼び出し側が付与する。
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

from src.collect.base import fetch

BASE = "https://keirin.kdreams.jp/gamboo/keirin-kaisai/race-card/odds"


def build_odds_url(kaisai_code: str, kaisai_day_code: str, race_no: int) -> str:
    """三連単オッズページのURLを組む。"""
    return f"{BASE}/{kaisai_code}/{kaisai_day_code}/{race_no}/3rentan/"


def parse_trifecta_odds(html: str) -> dict[tuple[int, int, int], float]:
    """三連単オッズページHTMLから {(1着,2着,3着): オッズ} を返す。"""
    soup = BeautifulSoup(html, "html.parser")
    out: dict[tuple[int, int, int], float] = {}

    for table in soup.select("table.odds_table.bt5"):
        classes = table.get("class", [])
        # 第3クラストークンが1着車番（例 ["odds_table","bt5","1"]）
        first = next((int(c) for c in classes if c.isdigit()), None)
        if first is None:
            continue

        rows = table.find_all("tr")
        # 列ヘッダ（3着車番）: th のうち数字テキストのものを列として拾う（前後の空thは除外）。
        # 選手名ヘッダ行（"1田 …"等）は数字onlyでないため2つ以上の純数字thが並ぶ行だけ採用。
        third_cars: list[int] = []
        header_idx = None
        for i, tr in enumerate(rows):
            nums = [th.get_text(strip=True) for th in tr.find_all("th")
                    if th.get_text(strip=True).isdigit()]
            if len(nums) >= 2:
                third_cars = [int(n) for n in nums]
                header_idx = i
                break
        if header_idx is None:
            continue

        # データ行: 先頭 th が2着車番、続く td が各3着列のオッズ
        for tr in rows[header_idx + 1:]:
            head = tr.find("th")
            if head is None or not head.get_text(strip=True).isdigit():
                continue
            second = int(head.get_text(strip=True))
            tds = tr.find_all("td")
            for col, td in enumerate(tds):
                if col >= len(third_cars):
                    break
                cls = td.get("class", [])
                if "empty" in cls:
                    continue
                txt = td.get_text(strip=True).replace(",", "")
                try:
                    odds = float(txt)
                except ValueError:
                    continue          # 欠場等でオッズ非表示
                third = third_cars[col]
                if len({first, second, third}) == 3:   # 同一車番の混入を除外
                    out[(first, second, third)] = odds
    return out


def parse_deadline(html: str) -> str | None:
    """オッズページから投票締切時刻 "HH:MM" を返す（<dt>締切</dt><dd>16:25</dd>）。無ければNone。"""
    soup = BeautifulSoup(html, "html.parser")
    for dt in soup.find_all("dt"):
        if "締切" in dt.get_text():
            dd = dt.find_next_sibling("dd")
            if dd:
                m = re.search(r"\d{1,2}:\d{2}", dd.get_text())
                if m:
                    return m.group(0)
    return None


def fetch_trifecta_odds(
    kaisai_code: str, kaisai_day_code: str, race_no: int
) -> tuple[dict[tuple[int, int, int], float], str | None]:
    """オッズページを取得しパースする。戻り値: (オッズ, 締切時刻)。ネットワークアクセスあり。"""
    url = build_odds_url(kaisai_code, kaisai_day_code, race_no)
    res = fetch(url)
    return parse_trifecta_odds(res.text), parse_deadline(res.text)
