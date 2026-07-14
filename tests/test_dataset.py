"""バックフィル用データセットの収集＋SQLite保存のテスト（オフライン）。"""
from pathlib import Path

from src.collect import dataset
from src.collect.dataset import collect_race_dataset
from src.collect.gamboo_schedule import Kaisai
from db.repository import DatasetRepo

FX = Path(__file__).parent / "fixtures"


def _patch(monkeypatch, odds_fixture, result_fixture=None):
    odds_html = (FX / odds_fixture).read_text(encoding="utf-8")
    res_html = (FX / result_fixture).read_text(encoding="utf-8") if result_fixture else ""

    def fake_fetch(url, **k):
        class R:  # base.fetch は .text を持つオブジェクトを返す
            pass
        r = R()
        r.text = res_html if "/result/" in url else odds_html
        return r
    monkeypatch.setattr(dataset, "fetch", fake_fetch)


def test_collect_race_dataset_offline(monkeypatch):
    _patch(monkeypatch, "gamboo_trifecta_sample.html", "gamboo_result_sample.html")
    k = Kaisai("3520251228", "35202512280100", "35", True)
    ds = collect_race_dataset(k, 11)
    assert ds.field_size == 9
    assert len(ds.odds_final) == 9 * 8 * 7          # 確定オッズ全504点
    assert ds.missing_odds == []
    assert len(ds.results) == 9                       # 着順9行
    assert ds.payout is not None and ds.payout.combo == (3, 9, 6)
    assert ds.has_result is True


def test_require_girls_skips_result_fetch(monkeypatch):
    # 男子戦フィクスチャ + require_girls=True → 結果ページを取得せず早期return
    _patch(monkeypatch, "gamboo_racecard_7car.html")   # A級7車
    k = Kaisai("1120260714", "11202607140100", "11", True)
    ds = collect_race_dataset(k, 1, require_girls=True)
    assert ds.is_girls is False
    assert ds.has_result is False
    assert len(ds.odds_final) == 210                  # オッズは取得済み


def test_dataset_repo_roundtrip(monkeypatch):
    _patch(monkeypatch, "gamboo_trifecta_sample.html", "gamboo_result_sample.html")
    k = Kaisai("3520251228", "35202512280100", "35", True)
    ds = collect_race_dataset(k, 11)

    repo = DatasetRepo(":memory:")
    repo.save_race(ds.race_id, "2025-12-28", ds.venue_code, ds.race_no,
                   ds.is_girls, ds.deadline, ds.field_size)
    assert repo.save_entries(ds.race_id, ds.entries) == 9
    assert repo.save_results(ds.race_id, ds.results) == 9
    assert repo.save_odds_final(ds.race_id, ds.odds_final) == 504
    repo.save_payout(ds.race_id, ds.payout)

    assert repo.race_ids() == [ds.race_id]
    assert repo.count("odds_final_trifecta") == 504
    assert repo.count("payouts_trifecta") == 1
    # 再保存してもUPSERTで重複しない
    repo.save_odds_final(ds.race_id, ds.odds_final)
    assert repo.count("odds_final_trifecta") == 504
    repo.close()
