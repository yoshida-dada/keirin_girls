"""特徴量アセンブラのテスト（実出走表フィクスチャ）。"""
from pathlib import Path

from src.collect.gamboo_racecard import parse_race_card, parse_recent_form
from src.features.assembler import build_features, FEATURE_COLUMNS

FX = Path(__file__).parent / "fixtures"


def _real():
    html = (FX / "gamboo_racecard_7car.html").read_text(encoding="utf-8")
    return parse_race_card(html), parse_recent_form(html)


def test_matrix_shape_and_index():
    entries, recent = _real()
    df = build_features(entries, recent)
    assert list(df.index) == [1, 2, 3, 4, 5, 6, 7]     # 車番昇順
    # 学習用の全特徴量列が存在する
    for col in FEATURE_COLUMNS:
        assert col in df.columns, col


def test_as_of_values_present():
    entries, recent = _real()
    df = build_features(entries, recent)
    # 相対得点: 最強(3番)は rel_score_max=0
    assert df.loc[3, "rel_score_max"] == 0.0
    # レース全体はブロードキャスト（全行同値）
    assert df["race_mean_score"].nunique() == 1
    # 直近由来（勝率）が入っている
    assert df["win_rate"].notna().all()


def test_leg_flags_exclusive():
    entries, recent = _real()
    df = build_features(entries, recent)
    # 各行の脚質フラグ合計は0か1（複数typeに同時該当しない）
    flags = df[["is_escape", "is_dash", "is_closing", "is_mark"]].sum(axis=1)
    assert (flags <= 1).all()


def test_graceful_without_recent():
    entries, _ = _real()
    df = build_features(entries)                 # recentなし
    assert df["racing_score"].notna().all()       # 出走表由来は出る
    assert df["win_rate"].isna().all()            # 直近由来はNaN
