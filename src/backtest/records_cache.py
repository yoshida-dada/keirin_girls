"""バックテスト用 ComboRecord のビルド＆キャッシュ（並列調査で再学習を避けるため）。

31特徴lambdarankを train期間で学習し、検証期間(out-of-sample)の全レース×210点の ComboRecord を
作ってpickleに保存/読込する。複数の分析スクリプト（サブエージェント）が同一の records を共有できる。

  from src.backtest.records_cache import get_records
  records = get_records("data/keirin.sqlite")   # 初回のみビルド(~数分)、以降はキャッシュ即読込
"""
from __future__ import annotations

import pickle
from pathlib import Path

_CACHE = Path(__file__).resolve().parents[2] / "data" / "_bt_records.pkl"


def get_records(db_path: str = "data/keirin.sqlite", test_frac: float = 0.35,
                rebuild: bool = False):
    """検証期間の ComboRecord リストを返す（キャッシュ優先）。"""
    if _CACHE.exists() and not rebuild:
        with open(_CACHE, "rb") as f:
            return pickle.load(f)
    from src.model.training_data import load_samples, PL_FEATURES_FULL
    from src.model.train_gbdt import train_gbdt
    from src.model.feature_augment import augment_samples
    from src.features.tactics_features import TACTIC_NAMES
    from src.model.evaluate import time_split
    from src.backtest.bucket_analysis import build_records

    base = load_samples(db_path, features=PL_FEATURES_FULL)
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    samples = augment_samples(base, db_path, feats31)
    train, test = time_split(samples, test_frac)
    model = train_gbdt(train)
    records = build_records(db_path, model, [s.race_id for s in test])
    _CACHE.parent.mkdir(exist_ok=True)
    with open(_CACHE, "wb") as f:
        pickle.dump(records, f)
    return records


if __name__ == "__main__":
    import sys
    recs = get_records(rebuild="--rebuild" in sys.argv)
    print(f"records: {len(recs)} / races: {len(set(r.race_id for r in recs))} / cache: {_CACHE}")
