"""バケット分析（Phase4）の集計ロジックのテスト（合成ComboRecordで検証）。"""
from src.backtest.bucket_analysis import ComboRecord, bucket_roi, odds_bucket_roi


def _rec(rtype, bucket, combo, p, odds, is_win, payout=0):
    return ComboRecord(race_id="R", race_type=rtype, odds_bucket=bucket, combo=combo,
                       model_prob=p, odds=odds, ev=p * odds, is_win=is_win, payout=payout)


def test_bucket_roi_basic():
    # 軸堅×50-100: 2点購入(EV>=1.1)、1点的中で払戻8000円(100円あたり)
    recs = [
        _rec("軸堅", "50-100", (1, 2, 3), 0.02, 60, True, payout=8000),   # EV=1.2 買い・的中
        _rec("軸堅", "50-100", (1, 3, 2), 0.02, 60, False),               # EV=1.2 買い・外れ
        _rec("軸堅", "50-100", (7, 6, 5), 0.001, 60, False),              # EV=0.06 買わない
    ]
    res = bucket_roi(recs, ev_threshold=1.1, min_prob=0.0, max_odds=None)
    b = res[("軸堅", "50-100")]
    assert b["n_bets"] == 2               # EV>=1.1 は2点
    assert b["n_hits"] == 1
    assert b["stake"] == 200              # 2点×100円
    assert b["ret"] == 8000
    assert b["roi"] == 40.0               # 8000/200
    assert b["n_all"] == 3                # 較正材料は全買い目


def test_min_prob_and_max_odds_guards():
    recs = [
        _rec("標準", "300+", (1, 2, 3), 0.001, 2000, True, payout=200000),  # 低確率×高オッズ
    ]
    # ガード無し: 買う
    assert bucket_roi(recs, 1.1, min_prob=0.0, max_odds=None)[("標準", "300+")]["n_bets"] == 1
    # min_prob ガード: 除外
    assert bucket_roi(recs, 1.1, min_prob=0.005, max_odds=None)[("標準", "300+")]["n_bets"] == 0
    # max_odds ガード: 除外
    assert bucket_roi(recs, 1.1, min_prob=0.0, max_odds=500)[("標準", "300+")]["n_bets"] == 0


def test_odds_bucket_roi_favorite_longshot():
    # 本命帯は的中で回収、穴帯は全外れ → ROIに差
    recs = [
        _rec("軸堅", "0-10", (1, 2, 3), 0.15, 6, True, payout=600),
        _rec("軸堅", "300+", (7, 6, 5), 0.001, 400, False),
        _rec("軸堅", "300+", (6, 7, 5), 0.001, 400, False),
    ]
    res = odds_bucket_roi(recs, ev_threshold=0.0)
    assert res["0-10"]["roi"] == 6.0           # 600/100
    assert res["300+"]["roi"] == 0.0           # 全外れ
    assert res["300+"]["n"] == 2


def test_calibration_recorded():
    recs = [_rec("混戦", "100-300", (1, 2, 3), 0.5, 150, True, payout=15000),
            _rec("混戦", "100-300", (3, 2, 1), 0.5, 150, False)]
    b = bucket_roi(recs, 1.1)[("混戦", "100-300")]
    assert b["brier"] is not None and 0.0 <= b["brier"] <= 1.0
    assert b["ece"] is not None
