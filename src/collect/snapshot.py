"""オッズ時系列スナップショットの収集ロジック（S1・課題D/haircut推定の前提）。

配布元が無いため自作する。締切60分前〜締切までの窓で、各レースのオッズページを5〜10分刻みで
取得し、取得時刻付きで全点(7車=210点)をDBに蓄積する。ここは1回分の取得と窓判定を担い、
定期実行(スケジューラ)は scripts/collect_odds_snapshot.py が担う。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from db.repository import SnapshotRepo
from src.collect.base import detect_missing_trifecta, fetch
from src.collect.gamboo_odds import build_odds_url, parse_trifecta_odds, parse_deadline
from src.collect.gamboo_racecard import parse_race_card, is_girls_race
from src.collect.gamboo_schedule import Kaisai


def _fetch_page(kaisai_code: str, kaisai_day_code: str, race_no: int) -> str:
    """三連単オッズページのHTMLを1回取得する（オッズ＋締切＋出走表を同梱）。"""
    return fetch(build_odds_url(kaisai_code, kaisai_day_code, race_no)).text


def build_race_id(kaisai_day_code: str, race_no: int) -> str:
    """スナップショットのrace_id。開催日コード + レース番号2桁で一意。"""
    return f"{kaisai_day_code}{race_no:02d}"


def parse_deadline_dt(deadline_hhmm: str, ref: datetime) -> datetime:
    """"HH:MM" を ref と同日の datetime に。締切が翌0時台へ跨ぐミッドナイトは翌日扱い。"""
    h, m = (int(x) for x in deadline_hhmm.split(":"))
    dt = ref.replace(hour=h, minute=m, second=0, microsecond=0)
    if dt < ref - timedelta(hours=12):     # 深夜帯の日跨ぎ補正
        dt += timedelta(days=1)
    return dt


def in_collection_window(
    deadline_hhmm: str, now: datetime, lead_min: int = 60
) -> bool:
    """now が「締切 lead_min 分前 〜 締切」の収集窓内か。"""
    deadline = parse_deadline_dt(deadline_hhmm, now)
    return (deadline - timedelta(minutes=lead_min)) <= now <= deadline


def collect_race_snapshot(
    kaisai: Kaisai,
    race_no: int,
    repo: SnapshotRepo,
    now: datetime | None = None,
    only_in_window: bool = False,
    only_girls: bool = True,
    lead_min: int = 60,
) -> dict:
    """1レースの現時点オッズを取得しスナップショットとして保存する。

    オッズページ1回の取得で オッズ＋締切＋出走表 をまとめて解析する。
    only_girls=True のとき、L級（ガールズ）でないレースは保存せずスキップ（判定はレース単位）。
    only_in_window=True のとき、締切 lead_min 分前〜締切の窓外なら保存せずスキップ。
    戻り値: {race_id, n_odds, field_size, deadline, missing, saved, in_window, is_girls}。
    """
    now = now or datetime.now()
    html = _fetch_page(kaisai.kaisai_code, kaisai.kaisai_day_code, race_no)
    odds = parse_trifecta_odds(html)
    deadline = parse_deadline(html)
    entries = parse_race_card(html)
    girls = is_girls_race(entries)

    field_size = max((max(c) for c in odds), default=0)
    missing = detect_missing_trifecta(odds, field_size) if field_size else []
    race_id = build_race_id(kaisai.kaisai_day_code, race_no)

    in_window = bool(deadline) and in_collection_window(deadline, now, lead_min)
    should_save = (
        bool(odds)
        and (girls or not only_girls)
        and (in_window or not only_in_window)
    )
    saved = repo.save_snapshot(race_id, odds, now) if should_save else 0
    return {
        "race_id": race_id, "n_odds": len(odds), "field_size": field_size,
        "deadline": deadline, "missing": missing, "saved": saved,
        "in_window": in_window, "is_girls": girls,
    }
