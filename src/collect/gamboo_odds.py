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


def build_exacta_odds_url(kaisai_code: str, kaisai_day_code: str, race_no: int) -> str:
    """二車単オッズページのURLを組む（スラッグは 2shatan、2026-08調査で確認）。"""
    return f"{BASE}/{kaisai_code}/{kaisai_day_code}/{race_no}/2shatan/"


def parse_exacta_odds(html: str) -> dict[tuple[int, int], float]:
    """二車単オッズページHTMLから {(1着,2着): オッズ} を返す。

    構造（2026-08調査）: 単一 <table class="odds_table">。先頭行の th.n{k}=2着車番（列）、
    データ行の先頭 th.n{k}=1着車番、続く td が各2着列のオッズ。td.empty は対角(同一車=不可)、
    "9999.9" は非発売のセンチネルなので除外する。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.odds_table")
    if table is None:
        return {}
    rows = table.find_all("tr")
    # 列ヘッダ（2着車番）: th.n{digit} の数字テキストを列順に
    second_cars: list[int] = []
    for tr in rows:
        ths = [th for th in tr.find_all("th")
               if th.get_text(strip=True).isdigit()
               and any(c.startswith("n") for c in th.get("class", []))]
        if len(ths) >= 2:
            second_cars = [int(th.get_text(strip=True)) for th in ths]
            break
    if not second_cars:
        return {}

    out: dict[tuple[int, int], float] = {}
    for tr in rows:
        head = tr.find("th")
        if head is None or not head.get_text(strip=True).isdigit():
            continue
        if not any(c.startswith("n") for c in head.get("class", [])):
            continue
        first = int(head.get_text(strip=True))
        tds = tr.find_all("td", recursive=False) or tr.find_all("td")
        for col, td in enumerate(tds):
            if col >= len(second_cars):
                break
            if "empty" in td.get("class", []):
                continue
            txt = td.get_text(strip=True).replace(",", "")
            try:
                odds = float(txt)
            except ValueError:
                continue
            if odds >= 9999:                       # 非発売センチネル
                continue
            second = second_cars[col]
            if first != second:
                out[(first, second)] = odds
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


# 開催格の表記ゆれ対策。h1/title に入る全角表記を第一情報源にする。
# アイコンのCSSクラス(gr1..gr5)はフォールバック: 特殊開催（全プロ記念）で実態とズレる例を確認済み。
_GRADE_TEXT_RE = re.compile(r"(ＧＰ|Ｇ[１２３]|Ｆ[１２])")
_Z2H = str.maketrans("ＧＦＰ１２３", "GFP123")
_GRADE_BY_CLASS = {"gr1": "F2", "gr2": "F1", "gr3": "G3", "gr4": "G2",
                   "gr5": "G1", "gr6": "GP"}   # gr6は2025年末の平塚GPで実測


def parse_race_meta(html: str) -> dict:
    """オッズページから 格/会場/開催名/レース名 を返す。取れない項目は None。

    オッズページは既に取得済みなので**追加フェッチ0**でグレードが付く。
    レース名（"Ｓ級決勝"/"二次予選"等）は語彙が開催格で変わるため生文字列で返し、
    正規化は表示側に任せる。
    """
    soup = BeautifulSoup(html, "html.parser")
    grade = None
    h1 = soup.find("h1", class_="section_title")          # "出走表詳細 松山Ｇ１ レース情報"
    if h1:
        m = _GRADE_TEXT_RE.search(h1.get_text())
        if m:
            grade = m.group(1).translate(_Z2H)
    if grade is None:
        ic = soup.find("span", class_="icon_grade")
        if ic:
            grade = next((_GRADE_BY_CLASS[c] for c in ic.get("class", [])
                          if c in _GRADE_BY_CLASS), None)
    h2 = soup.find("h2", class_="title")                   # 会場 + 開催名
    vd = h2.find("span", class_="velodrome") if h2 else None
    rc = h2.find("span", class_="race") if h2 else None
    st = soup.select_one("div.race_title_header p.status")  # レース名
    return {
        "grade": grade,
        "venue": vd.get_text(strip=True) if vd else None,
        "meet_name": rc.get_text(strip=True) if rc else None,
        "race_name": st.get_text(strip=True) if st else None,
    }


def fetch_trifecta_odds(
    kaisai_code: str, kaisai_day_code: str, race_no: int
) -> tuple[dict[tuple[int, int, int], float], str | None]:
    """オッズページを取得しパースする。戻り値: (オッズ, 締切時刻)。ネットワークアクセスあり。"""
    url = build_odds_url(kaisai_code, kaisai_day_code, race_no)
    res = fetch(url)
    return parse_trifecta_odds(res.text), parse_deadline(res.text)


def fetch_exacta_odds(
    kaisai_code: str, kaisai_day_code: str, race_no: int
) -> dict[tuple[int, int], float]:
    """二車単オッズページを取得しパースする。戻り値: {(1着,2着): オッズ}。ネットワークあり。"""
    url = build_exacta_odds_url(kaisai_code, kaisai_day_code, race_no)
    return parse_exacta_odds(fetch(url).text)
