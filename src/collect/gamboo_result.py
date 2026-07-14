"""GambooBET レース結果パーサ（S1・学習ラベル＋バックテストの前提）。

URL: https://keirin.kdreams.jp/gamboo/keirin-kaisai/race-card/result/{開催}/{開催日}/{R}/
  result_table : 着順・車番・選手名・着差・上り(上がりタイム)・決まり手・S/B・勝敗因
  refund_table : 全券種の払戻（枠連/車連/3連複/ワイド/枠単/車単/三連単）＋人気

三連単の払戻（payouts_trifecta）はバックテストのROI決済に、着順は学習ラベルに使う。
バケット分析の「全210点機械買い」には別途 odds_final_trifecta（オッズページの確定オッズ）が必要
（refund_tableは的中1点のみ）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from src.collect.base import fetch
from src.collect.gamboo_odds import build_odds_url  # URL規則の流用


@dataclass
class ResultRow:
    position: int | None       # 着順（失格/欠は None）
    car_number: int
    rider_name: str
    margin: str                # 着差
    last_lap: float | None     # 上り＝上がりタイム(秒)
    kimarite: str              # 決まり手（1着以外は空のことが多い）
    sb: str                    # S / B マーク
    comment: str               # 勝敗因


@dataclass
class TrifectaPayout:
    combo: tuple[int, int, int]   # (1着,2着,3着)
    payout: int                   # 100円あたり払戻金（円）
    popularity: int | None        # 人気順位


def build_result_url(kaisai_code: str, kaisai_day_code: str, race_no: int) -> str:
    return build_odds_url(kaisai_code, kaisai_day_code, race_no).replace(
        "/race-card/odds/", "/race-card/result/").rsplit("3rentan/", 1)[0]


def _to_int(s: str) -> int | None:
    m = re.search(r"-?\d+", s or "")
    return int(m.group(0)) if m else None


def parse_results(html: str) -> list[ResultRow]:
    """着順テーブルから結果行（着順昇順）を返す。"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="result_table")
    if table is None:
        return []
    rows = table.find_all("tr")
    out: list[ResultRow] = []
    for tr in rows[1:]:                       # 先頭はヘッダ
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if len(cells) < 9:
            continue
        car = _to_int(cells[2])
        if car is None:
            continue
        out.append(ResultRow(
            position=_to_int(cells[1]), car_number=car, rider_name=cells[3],
            margin=cells[4],
            last_lap=(float(cells[5]) if re.match(r"^\d+\.\d+$", cells[5]) else None),
            kimarite=cells[6], sb=cells[7], comment=cells[8],
        ))
    return out


def parse_trifecta_payout(html: str) -> TrifectaPayout | None:
    """払戻テーブルから三連単の (組合せ, 払戻金, 人気) を返す。券種はセパレータで判定。

    三連単は "a-b-c 74,450円 (269)" 形式（"a=b=c" は三連複なので除外）。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="refund_table")
    text = table.get_text(" ", strip=True) if table else ""
    # a-b-c（ハイフン結合＝順序券種）＝三連単。金額と人気(かっこ)を取る。
    m = re.search(r"(\d+)-(\d+)-(\d+)\s*([\d,]+)\s*円\s*(?:\((\d+)\))?", text)
    if not m:
        return None
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    payout = int(m.group(4).replace(",", ""))
    pop = int(m.group(5)) if m.group(5) else None
    return TrifectaPayout(combo=(a, b, c), payout=payout, popularity=pop)


def fetch_result(kaisai_code: str, kaisai_day_code: str, race_no: int
                 ) -> tuple[list[ResultRow], TrifectaPayout | None]:
    """結果ページを取得しパースする。戻り値: (着順, 三連単払戻)。ネットワークアクセスあり。"""
    res = fetch(build_result_url(kaisai_code, kaisai_day_code, race_no))
    return parse_results(res.text), parse_trifecta_payout(res.text)
