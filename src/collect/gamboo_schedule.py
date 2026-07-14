"""GambooBET 開催一覧・レース一覧の発見スクレイパー（S1）。

収集対象URLを組むための最上流。日付 → その日の開催（会場×デイ/ナイト）一覧 →
各開催のレース一覧（レース番号・締切時刻）を辿り、オッズURL（gamboo_odds）へ渡す。

ページ構造（2026-07調査）:
  日付別開催一覧: /gamboo/kaisai/{YYYY}/{MM}/{DD}/
    <ul class="kaisai_list"> の直下 <li>（会場ごと）。ガールズ開催は <span class="icon_girls"> を含む。
    会場内に race-list リンク /race-card ... /race-list/{開催コード}/{開催日コード}/ が
    デイ/ナイト分（01/02）並ぶ。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from src.collect.base import fetch

HOST = "https://keirin.kdreams.jp"


@dataclass(frozen=True)
class Kaisai:
    kaisai_code: str        # 場コード(2) + 開催日YYYYMMDD（例 "1120260714"）
    kaisai_day_code: str    # kaisai_code + デイ/ナイト連番（例 "11202607140100"）
    venue_code: str         # 場コード（例 "11"）
    is_girls: bool          # ガールズ開催（icon_girls）


def build_kaisai_list_url(year: int, month: int, day: int) -> str:
    return f"{HOST}/gamboo/kaisai/{year:04d}/{month:02d}/{day:02d}/"


def parse_kaisai_list(html: str) -> list[Kaisai]:
    """日付別開催一覧HTMLから、その日の全開催（会場×デイ/ナイト）を返す。"""
    soup = BeautifulSoup(html, "html.parser")
    ul = soup.find("ul", class_="kaisai_list")
    if ul is None:
        return []
    out: list[Kaisai] = []
    for li in ul.find_all("li", recursive=False):
        is_girls = li.find("span", class_="icon_girls") is not None
        codes = {}   # kaisai_day_code -> kaisai_code（重複排除）
        for a in li.find_all("a", href=True):
            if "/race-list/" not in a["href"]:
                continue
            parts = a["href"].split("/race-list/")[1].strip("/").split("/")
            if len(parts) >= 2:
                codes[parts[1]] = parts[0]
        for day_code, code in codes.items():
            out.append(Kaisai(kaisai_code=code, kaisai_day_code=day_code,
                              venue_code=code[:2], is_girls=is_girls))
    return out


def kaisai_race_date(kaisai_day_code: str):
    """開催日コードから実施日を計算する。

    形式: {場2}{開催初日YYYYMMDD}{開催日目NN}{連番2}（例 62202607130200 = 場62・初日07-13・2日目）。
    実施日 = 初日 + (開催日目 - 1)日。/gamboo/kaisai/YYYY/MM/DD/ は開催中の全日程を返すため、
    特定日のレースだけ選ぶにはこの実施日でフィルタする（初日と2日目の混在を防ぐ）。
    """
    from datetime import date, timedelta
    start = date(int(kaisai_day_code[2:6]), int(kaisai_day_code[6:8]), int(kaisai_day_code[8:10]))
    nn = int(kaisai_day_code[10:12])
    return start + timedelta(days=nn - 1)


def build_race_list_url(kaisai_code: str, kaisai_day_code: str) -> str:
    return f"{HOST}/gamboo/keirin-kaisai/race-list/{kaisai_code}/{kaisai_day_code}/"


def parse_race_numbers(html: str, kaisai_code: str, kaisai_day_code: str) -> list[int]:
    """レース一覧HTMLから、その開催のレース番号一覧（昇順）を返す。

    race-card/odds リンク中の {開催}/{開催日}/{R}/ の R を集める。
    """
    pat = re.compile(
        rf"/race-card/(?:odds/)?{re.escape(kaisai_code)}/{re.escape(kaisai_day_code)}/(\d+)/"
    )
    nums = {int(m) for m in pat.findall(html)}
    return sorted(nums)


def fetch_girls_kaisai(year: int, month: int, day: int) -> list[Kaisai]:
    """指定日のガールズ開催のみを返す（ネットワークアクセスあり）。"""
    res = fetch(build_kaisai_list_url(year, month, day))
    return [k for k in parse_kaisai_list(res.text) if k.is_girls]


def fetch_race_numbers(kaisai: Kaisai) -> list[int]:
    """開催のレース番号一覧を取得する（ネットワークアクセスあり）。"""
    res = fetch(build_race_list_url(kaisai.kaisai_code, kaisai.kaisai_day_code))
    return parse_race_numbers(res.text, kaisai.kaisai_code, kaisai.kaisai_day_code)


# 級班の短縮表記（レース一覧の級班列）。L級＝ガールズ。
_RANK_RE = re.compile(r"^(SS|S[12]|A[123]|L[12])$")


def parse_girls_race_numbers(html: str, kaisai_code: str, kaisai_day_code: str) -> list[int]:
    """レース一覧HTMLから、ガールズ(L級)レースの番号だけを返す（昇順）。

    レース一覧ページは各レースの出走表テーブル（級班列付き）を12個持つ。各テーブルの直前の
    オッズリンクからレース番号を、級班セルからL級判定を取り、**オッズページを取らずに**
    ガールズ戦を絞り込む（男子戦の全スキャンを省く最大の高速化）。
    """
    soup = BeautifulSoup(html, "html.parser")
    odds_href = re.compile(
        rf"/odds/{re.escape(kaisai_code)}/{re.escape(kaisai_day_code)}/(\d+)/")
    girls: set[int] = set()
    for tb in soup.find_all("table"):
        if not tb.find("td", class_="num"):
            continue
        # 級班は「選手行(td.num を持つ行)」の級班セルだけを見る（他行の紛れを除外）。
        ranks = []
        for tr in tb.find_all("tr"):
            if tr.find("td", class_="num") is None:
                continue
            for c in tr.find_all("td"):
                txt = c.get_text(strip=True)
                if _RANK_RE.match(txt):
                    ranks.append(txt)
                    break
        if not ranks or not all(r.startswith("L") for r in ranks):
            continue
        a = tb.find_previous("a", href=odds_href)
        if a:
            girls.add(int(odds_href.search(a["href"]).group(1)))
    return sorted(girls)


def fetch_girls_race_numbers(kaisai: Kaisai) -> list[int]:
    """開催のガールズ(L級)レース番号のみを取得する（レース一覧1ページのみ・ネットワークあり）。"""
    res = fetch(build_race_list_url(kaisai.kaisai_code, kaisai.kaisai_day_code))
    return parse_girls_race_numbers(res.text, kaisai.kaisai_code, kaisai.kaisai_day_code)
