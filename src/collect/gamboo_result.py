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


@dataclass
class TrioPayout:
    """三連複（順不同）。combo は車番昇順。"""
    combo: tuple[int, int, int]
    payout: int
    popularity: int | None


@dataclass
class ExactaPayout:
    """二車単（車単・着順どおり）。combo は (1着, 2着)。"""
    combo: tuple[int, int]
    payout: int
    popularity: int | None


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


def parse_trio_payout(html: str) -> TrioPayout | None:
    """払戻テーブルから三連複の (組合せ, 払戻金, 人気) を返す。

    実物の払戻テーブル（2026-08-18 確認）は1行に全券種が並ぶ:
      ['2 枠 連','複','1=4 190円 (1)','2 車 連','複','1=5 180円 (1)',
       '3 連 勝','複','1=5=7 410円 (2)','ワ イ ド','1=5 140円 (1) ...']
    **`a=b=c`（イコール3つ＝順不同）が三連複**。`a=b`（2つ）は車連・枠連・ワイドで、
    ワイドは1セルに3組入るので、3連結だけを拾えば取り違えない。
    枠連も `1=4` 形式なので、**イコールが2個の組は一切拾わない**。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="refund_table")
    text = table.get_text(" ", strip=True) if table else ""
    m = re.search(r"(\d+)=(\d+)=(\d+)\s*([\d,]+)\s*円\s*(?:\((\d+)\))?", text)
    if not m:
        return None
    cars = tuple(sorted((int(m.group(1)), int(m.group(2)), int(m.group(3)))))
    return TrioPayout(combo=cars, payout=int(m.group(4).replace(",", "")),
                      popularity=int(m.group(5)) if m.group(5) else None)


def _iter_refund_entries(html: str) -> list[tuple[str, int, int | None]]:
    """払戻テーブルの各配当を (組合せ文字列, 払戻金, 人気) の並び順どおりのリストで返す。

    各配当は `<dl class="cf"><dt>組合せ</dt><dd>N円 <span>(人気)</span></dd></dl>` 構造
    （2026-08 確認）。券種はここでは判定せず、呼び出し側が dt のセパレータで振り分ける。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="refund_table")
    out: list[tuple[str, int, int | None]] = []
    if table is None:
        return out
    for dl in table.find_all("dl"):
        dt = dl.find("dt")
        dd = dl.find("dd")
        if dt is None or dd is None:
            continue
        combo = dt.get_text("", strip=True)
        dd_text = dd.get_text(" ", strip=True)
        m = re.search(r"([\d,]+)\s*円", dd_text)
        if not m:
            continue
        payout = int(m.group(1).replace(",", ""))
        pm = re.search(r"\((\d+)\)", dd_text)
        pop = int(pm.group(1)) if pm else None
        out.append((combo, payout, pop))
    return out


def parse_exacta_payout(html: str) -> ExactaPayout | None:
    """払戻テーブルから二車単（車単）の (組合せ, 払戻金, 人気) を返す。

    払戻テーブルの `a-b`（ハイフン・数字2つ）は **枠単と車単の2つ**が並ぶ（8・9車立て。
    2026-08 確認）。両者は組合せが違う（枠単＝枠番, 車単＝車番）ため、**三連単の1・2着**
    `(1着,2着)` と一致するハイフン2車組が車単。これで枠単を着順そのもので確実に外せる。
      * 7車立て: 枠番=車番なので枠単は出ず、車単1つ＝三連単の1・2着に一致する。
      * 2着同着など複数の車単があるレース: 三連単側で採った1着-2着に対応する1点を返す
        （三連単・三連複パーサが1組を返すのと揃える）。
    三連単が取れない場合のフォールバックとして、末尾のハイフン2車組（＝車単の位置）を採る。
    """
    tri = parse_trifecta_payout(html)
    target = (tri.combo[0], tri.combo[1]) if tri else None
    chosen: tuple[tuple[int, int], int, int | None] | None = None
    for c, pay, pp in _iter_refund_entries(html):
        m = re.fullmatch(r"(\d+)-(\d+)", c)      # ハイフン・数字ちょうど2つ（三連単 a-b-c は除外）
        if not m:
            continue
        pair = (int(m.group(1)), int(m.group(2)))
        if target is None:
            chosen = (pair, pay, pp)             # フォールバック: 最後のハイフン2車組＝車単
        elif pair == target:
            chosen = (pair, pay, pp)             # 三連単の1・2着に一致＝車単（枠単を除外）
    if chosen is None:
        return None
    return ExactaPayout(combo=chosen[0], payout=chosen[1], popularity=chosen[2])


def fetch_result(kaisai_code: str, kaisai_day_code: str, race_no: int
                 ) -> tuple[list[ResultRow], TrifectaPayout | None]:
    """結果ページを取得しパースする。戻り値: (着順, 三連単払戻)。ネットワークアクセスあり。"""
    res = fetch(build_result_url(kaisai_code, kaisai_day_code, race_no))
    return parse_results(res.text), parse_trifecta_payout(res.text)
