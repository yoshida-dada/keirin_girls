"""S1収集パイプラインのテスト（開催発見→締切→スナップショット保存）。全てオフライン。"""
from datetime import datetime
from pathlib import Path

import pytest

from src.collect.gamboo_schedule import parse_kaisai_list, parse_race_numbers
from src.collect.gamboo_odds import parse_trifecta_odds, parse_deadline
from src.collect import snapshot
from src.collect.gamboo_schedule import Kaisai
from db.repository import SnapshotRepo, combo_to_str, combo_from_str

FX = Path(__file__).parent / "fixtures"


def test_parse_kaisai_list_flags_girls():
    html = (FX / "gamboo_kaisai_list.html").read_text(encoding="utf-8")
    kaisai = parse_kaisai_list(html)
    assert len(kaisai) >= 8
    girls = {k.venue_code for k in kaisai if k.is_girls}
    men = {k.venue_code for k in kaisai if not k.is_girls}
    assert girls == {"85", "11", "56", "62"}      # 調査で確認した当日ガールズ開催
    assert men == {"24", "61", "87", "27"}
    # コード整合: kaisai_day_code は kaisai_code で始まる
    for k in kaisai:
        assert k.kaisai_day_code.startswith(k.kaisai_code)
        assert k.venue_code == k.kaisai_code[:2]


def test_parse_race_numbers():
    html = (FX / "gamboo_race_list.html").read_text(encoding="utf-8")
    nums = parse_race_numbers(html, "1120260714", "11202607140100")
    assert nums == list(range(1, 13))             # 12レース


def test_parse_girls_race_numbers():
    # レース一覧の級班列からL級レースだけ絞り込む（男子戦のオッズ取得を省く高速化）。
    from src.collect.gamboo_schedule import parse_girls_race_numbers
    html = (FX / "gamboo_race_list.html").read_text(encoding="utf-8")
    girls = parse_girls_race_numbers(html, "1120260714", "11202607140100")
    assert girls == [6, 7]                          # 会場11の当日L級はR6/R7のみ


def test_parse_deadline():
    html = (FX / "gamboo_trifecta_sample.html").read_text(encoding="utf-8")
    assert parse_deadline(html) == "16:25"        # フィクスチャ実測


def test_combo_str_roundtrip():
    assert combo_to_str((3, 1, 5)) == "3-1-5"
    assert combo_from_str("3-1-5") == (3, 1, 5)


def test_repository_roundtrip():
    repo = SnapshotRepo(":memory:")
    odds = {(1, 2, 3): 67.4, (1, 3, 2): 70.2}
    t1 = datetime(2026, 7, 14, 16, 0, 0)
    t2 = datetime(2026, 7, 14, 16, 10, 0)
    assert repo.save_snapshot("R1", odds, t1) == 2
    repo.save_snapshot("R1", {(1, 2, 3): 60.0, (1, 3, 2): 71.0}, t2)   # 別時刻
    times = repo.snapshot_times("R1")
    assert len(times) == 2                          # 時系列2点
    assert repo.load_snapshot("R1", times[0])[(1, 2, 3)] == 67.4
    assert repo.load_snapshot("R1", times[1])[(1, 2, 3)] == 60.0       # 締切に向けオッズ下落
    repo.close()


def test_in_collection_window():
    now = datetime(2026, 7, 14, 16, 0, 0)
    assert snapshot.in_collection_window("16:25", now, lead_min=60)     # 締切25分前=窓内
    assert not snapshot.in_collection_window("18:00", now, lead_min=60) # 締切2時間前=窓外
    assert not snapshot.in_collection_window("15:55", now, lead_min=60) # 締切超過=窓外


def test_collect_race_snapshot_offline(monkeypatch):
    # 9車の男子戦フィクスチャ。オッズページ1枚を返すよう _fetch_page を差し替える。
    html = (FX / "gamboo_trifecta_sample.html").read_text(encoding="utf-8")
    monkeypatch.setattr(snapshot, "_fetch_page", lambda *a, **k: html)
    repo = SnapshotRepo(":memory:")
    kaisai = Kaisai("1120260714", "11202607140100", "11", True)
    now = datetime(2026, 7, 14, 16, 0, 0)

    res = snapshot.collect_race_snapshot(kaisai, 11, repo, now=now, only_girls=False)
    assert res["race_id"] == "11202607140100" + "11"   # kaisai_day_code + R番号2桁
    assert res["field_size"] == 9                  # フィクスチャは9車
    assert res["n_odds"] == 9 * 8 * 7              # 504点
    assert res["missing"] == []
    assert res["deadline"] == "16:25"
    assert res["is_girls"] is False                # A級=男子戦
    assert res["saved"] == 504
    assert len(repo.snapshot_times(res["race_id"])) == 1
    repo.close()


def test_collect_skips_non_girls_when_only_girls(monkeypatch):
    # only_girls=True の既定では、L級でないレースは保存しない（ガールズ判定はレース単位）。
    html = (FX / "gamboo_racecard_7car.html").read_text(encoding="utf-8")
    monkeypatch.setattr(snapshot, "_fetch_page", lambda *a, **k: html)
    repo = SnapshotRepo(":memory:")
    kaisai = Kaisai("1120260714", "11202607140100", "11", True)
    now = datetime(2026, 7, 14, 16, 0, 0)

    res = snapshot.collect_race_snapshot(kaisai, 1, repo, now=now, only_girls=True)
    assert res["is_girls"] is False
    assert res["n_odds"] == 210                    # 7車=210点は取得できている
    assert res["saved"] == 0                        # だが男子戦なので保存しない
    repo.close()
