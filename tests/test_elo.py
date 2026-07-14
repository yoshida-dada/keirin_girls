"""Eloレーティングのテスト（合成DBで as-of と勝者上昇を確認）。"""
import sqlite3

from src.model.elo import compute_pre_race_elo, final_elo_state, DEFAULT_ELO
from db.repository import DatasetRepo


def _mk_db(tmp_path):
    """車1が常に1着・車2が常に2着…の3レースを作る。"""
    db = tmp_path / "elo.sqlite"
    repo = DatasetRepo(db)
    from types import SimpleNamespace
    for i, rid in enumerate(["R1", "R2", "R3"]):
        repo.save_race(rid, f"2025-01-0{i+1}", "62", i + 1, True, None, 7)
        ents = [SimpleNamespace(car_number=c, bracket_number=c, rider_name=f"rider{c}",
                                prefecture="東京", age=25, term=120, class_rank="L1",
                                leg_type="逃", gear_ratio=3.9, racing_score=55.0)
                for c in range(1, 8)]
        repo.save_entries(rid, ents)
        results = [SimpleNamespace(position=c, car_number=c, rider_name=f"rider{c}",
                                   margin="", last_lap=11.0, kimarite="逃", sb="")
                   for c in range(1, 8)]
        repo.save_results(rid, results)
    repo.close()
    return db


def test_winner_gains_elo(tmp_path):
    db = _mk_db(tmp_path)
    final = final_elo_state(db)
    # 常に1着の rider1 は最高、常に最下位の rider7 は最低
    assert final["rider1"] > DEFAULT_ELO > final["rider7"]
    assert final["rider1"] == max(final.values())
    assert final["rider7"] == min(final.values())


def test_pre_race_is_as_of(tmp_path):
    db = _mk_db(tmp_path)
    pre = compute_pre_race_elo(db)
    # R1 の発走前は全員デフォルト（履歴なし）
    assert pre[("R1", 1)] == DEFAULT_ELO
    # R3 の発走前は rider1 が既に上昇している（R1,R2の結果が反映）
    assert pre[("R3", 1)] > DEFAULT_ELO
    # 発走前値なので、そのレースの結果は含まない（R2発走前 < R3発走前）
    assert pre[("R2", 1)] < pre[("R3", 1)]
