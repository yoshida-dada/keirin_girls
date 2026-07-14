"""GambooBET 出走表パーサ（S1 / S2・S3の入力）。

三連単オッズページ（race-card/odds/.../3rentan/）に同梱される racecard_table[0] から
各選手の 車番・枠番・選手名・府県・年齢・期別・級班・脚質・ギヤ倍数・競走得点 を抽出する。
1回のオッズページ取得で「オッズ＋締切＋出走表」が揃う。

★重要: ガールズ判定はレース単位。開催に icon_girls が付いていても個々のレースは男子戦のことがある
（例 2026-07-14 会場11 R1 は A級の男子戦）。級班が L で始まる行がガールズ（L級）。

登録番号はこのページに無いため rider_id は未取得。当面は (rider_name, term) で選手を識別し、
選手成績（競輪ステーション等）と突合する段階で登録番号を付与する。
"""
from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup

from src.collect.base import fetch
from src.collect.gamboo_odds import build_odds_url

# 脚質の正規化（マ→マーク。逃/捲/差はそのまま。男子の 追/両 等は原文保持）
_LEG_NORMALIZE = {"マ": "マーク"}


@dataclass
class Entry:
    car_number: int
    bracket_number: int | None
    rider_name: str
    prefecture: str
    age: int | None
    term: int | None            # 期別
    class_rank: str             # 級班 例 "L1"(ガールズ) / "A1"(男子)
    leg_type: str               # 脚質 逃/捲/差/マーク（男子は 追/両 等もあり）
    gear_ratio: float | None    # ギヤ倍数
    racing_score: float | None  # 競走得点


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _to_int(s: str) -> int | None:
    return int(s) if s and s.isdigit() else None


def parse_race_card(html: str) -> list[Entry]:
    """出走表HTMLから出走選手一覧（車番昇順）を返す。"""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.racecard_table")
    if table is None:
        return []

    entries: list[Entry] = []
    for num_cell in table.select("td.num"):
        row = num_cell.find_parent("tr")
        if row is None:
            continue
        car = _to_int(num_cell.get_text(strip=True))
        if car is None:
            continue
        bracket_cell = row.find("td", class_="bracket")
        bracket = _to_int(bracket_cell.get_text(strip=True)) if bracket_cell else None

        rider_cell = row.find("td", class_="rider")
        if rider_cell is None:
            continue
        home = rider_cell.find("span", class_="home")
        home_txt = home.get_text(strip=True) if home else ""
        name = rider_cell.get_text(" ", strip=True)
        if home_txt:
            name = name.replace(home_txt, "").strip()
        prefecture, age, term = "", None, None
        parts = home_txt.split("/")
        if len(parts) == 3:
            prefecture = parts[0].replace("　", "")   # 全角スペース除去
            age, term = _to_int(parts[1]), _to_int(parts[2])

        # 選手名セルの直後4つ = 級班, 脚質, ギヤ倍数, 競走得点
        after = rider_cell.find_next_siblings("td")
        vals = [c.get_text(strip=True) for c in after[:4]]
        vals += [""] * (4 - len(vals))
        class_rank, leg_raw, gear, score = vals[0], vals[1], vals[2], vals[3]
        leg_type = _LEG_NORMALIZE.get(leg_raw, leg_raw)

        entries.append(Entry(
            car_number=car, bracket_number=bracket, rider_name=name,
            prefecture=prefecture, age=age, term=term, class_rank=class_rank,
            leg_type=leg_type, gear_ratio=_to_float(gear), racing_score=_to_float(score),
        ))
    entries.sort(key=lambda e: e.car_number)
    return entries


@dataclass
class RecentForm:
    """出走表に同梱される「直近4ヶ月成績」（as-of＝そのレース発走前に確定していた値）。

    S2のリークセーフな第一次ソース（docs/design_s2_features.md）。勝率系は 0-1 の分数で保持。
    逃/捲/差/マは決まり手（勝ち脚）の回数。
    """
    car_number: int
    s_count: int | None = None      # S（スタート）回数
    b_count: int | None = None      # B（バック先頭）回数
    escape: int | None = None       # 逃
    dash: int | None = None         # 捲
    closing: int | None = None      # 差
    mark: int | None = None         # マ
    first: int | None = None
    second: int | None = None
    third: int | None = None
    out: int | None = None          # 着外
    win_rate: float | None = None   # 勝率(0-1)
    top2_rate: float | None = None  # 2連対率
    top3_rate: float | None = None  # 3連対率

    @property
    def starts(self) -> int | None:
        vals = [self.first, self.second, self.third, self.out]
        return sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None


def parse_recent_form(html: str) -> dict[int, RecentForm]:
    """出走表の racecard_table[0] から各選手の直近4ヶ月成績を {車番: RecentForm} で返す。

    選手名セルの直後は 級班/脚質/ギヤ/競走得点 の4セル、その後に
    S/B/逃/捲/差/マ/1着/2着/3着/着外/勝率/2連対率/3連対率 の13セルが続く（枠番欠落に非依存）。
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.racecard_table")
    if table is None:
        return {}

    out: dict[int, RecentForm] = {}
    for num_cell in table.select("td.num"):
        car = _to_int(num_cell.get_text(strip=True))
        row = num_cell.find_parent("tr")
        if car is None or row is None:
            continue
        rider_cell = row.find("td", class_="rider")
        if rider_cell is None:
            continue
        after = rider_cell.find_next_siblings("td")
        if len(after) < 17:                    # 直近4ヶ月列が無い表（table[1]等）はスキップ
            continue
        vals = [c.get_text(strip=True) for c in after[4:17]]   # 得点の後ろ13列

        def pct(s: str) -> float | None:
            v = _to_float(s)
            return v / 100.0 if v is not None else None

        out[car] = RecentForm(
            car_number=car,
            s_count=_to_int(vals[0]), b_count=_to_int(vals[1]),
            escape=_to_int(vals[2]), dash=_to_int(vals[3]),
            closing=_to_int(vals[4]), mark=_to_int(vals[5]),
            first=_to_int(vals[6]), second=_to_int(vals[7]),
            third=_to_int(vals[8]), out=_to_int(vals[9]),
            win_rate=pct(vals[10]), top2_rate=pct(vals[11]), top3_rate=pct(vals[12]),
        )
    return out


def is_girls_race(entries: list[Entry]) -> bool:
    """出走表がガールズ（L級）レースか。全員の級班が L で始まればガールズ。"""
    ranks = [e.class_rank for e in entries if e.class_rank]
    return bool(ranks) and all(r.startswith("L") for r in ranks)


def fetch_race_card(kaisai_code: str, kaisai_day_code: str, race_no: int) -> list[Entry]:
    """出走表を取得しパースする（オッズページ同梱。ネットワークアクセスあり）。"""
    res = fetch(build_odds_url(kaisai_code, kaisai_day_code, race_no))
    return parse_race_card(res.text)
