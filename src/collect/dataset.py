"""過去レースのデータセット収集（S1バックフィル）。1レース分をまとめて取得する。

学習ラベル＋バックテストのため、確定後の1レースについて次を取得:
  出走表(entries) / 着順(results) / 三連単払戻(payout) / 確定オッズ全210点(odds_final)
オッズページ1回＋結果ページ1回の計2フェッチ（出走表はオッズページに同梱）。
選手成績の名前解決はコストが高いのでバックフィルの本ループには含めず、別パスで付与する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.collect.base import fetch, detect_missing_trifecta
from src.collect.gamboo_odds import build_odds_url, parse_trifecta_odds, parse_deadline
from src.collect.gamboo_racecard import parse_race_card, parse_recent_form, is_girls_race, Entry
from src.collect.gamboo_result import build_result_url, parse_results, parse_trifecta_payout
from src.collect.gamboo_schedule import Kaisai
from src.collect.snapshot import build_race_id


@dataclass
class RaceDataset:
    race_id: str
    venue_code: str
    race_no: int
    is_girls: bool
    deadline: str | None
    field_size: int
    entries: list[Entry]
    results: list
    payout: object | None          # TrifectaPayout
    odds_final: dict               # {(a,b,c): odds} 全210点
    missing_odds: list             # 欠損している組合せ
    has_result: bool
    recent: dict = field(default_factory=dict)   # {車番: RecentForm} 直近4ヶ月as-osスタッツ


def collect_race_dataset(kaisai: Kaisai, race_no: int,
                         require_girls: bool = False) -> RaceDataset:
    """1レースの出走表＋確定オッズ（オッズページ）と着順＋払戻（結果ページ）を取得する。

    require_girls=True のとき、L級でないレースは結果ページを取得せず早期に返す（無駄取得の回避）。
    """
    # オッズページ（出走表＋確定オッズ＋締切を同梱）
    odds_html = fetch(build_odds_url(kaisai.kaisai_code, kaisai.kaisai_day_code, race_no)).text
    entries = parse_race_card(odds_html)
    odds = parse_trifecta_odds(odds_html)
    deadline = parse_deadline(odds_html)
    recent = parse_recent_form(odds_html)          # 直近4ヶ月as-osスタッツ（同梱）
    field_size = max((max(c) for c in odds), default=len(entries))
    missing = detect_missing_trifecta(odds, field_size) if field_size else []
    girls = is_girls_race(entries)

    results, payout = [], None
    if require_girls and not girls:
        return RaceDataset(
            race_id=build_race_id(kaisai.kaisai_day_code, race_no),
            venue_code=kaisai.venue_code, race_no=race_no, is_girls=False,
            deadline=deadline, field_size=field_size, entries=entries, results=[],
            payout=None, odds_final=odds, missing_odds=missing, has_result=False,
            recent=recent)

    # 結果ページ（着順＋三連単払戻）。未確定なら空。
    try:
        res_html = fetch(build_result_url(kaisai.kaisai_code, kaisai.kaisai_day_code, race_no)).text
        results = parse_results(res_html)
        payout = parse_trifecta_payout(res_html)
    except Exception:
        pass

    return RaceDataset(
        race_id=build_race_id(kaisai.kaisai_day_code, race_no),
        venue_code=kaisai.venue_code, race_no=race_no,
        is_girls=girls, deadline=deadline, field_size=field_size,
        entries=entries, results=results, payout=payout,
        odds_final=odds, missing_odds=missing, has_result=bool(results),
        recent=recent,
    )
