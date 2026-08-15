"""学習済みモデルの保存・読込＋出走選手からの推論ヘルパー（S3運用連携）。

API/ダッシュボードが学習済みモデルで確率を出せるよう、モデル成果物の入出力と
「出走選手(Entry) → 1着強さ・三連単210通り確率」の一貫した推論経路を提供する。
特徴量の組み立て（assembler＋PL_FEATURES）をここに閉じ込め、呼び出し側は Entry を渡すだけにする。
"""
from __future__ import annotations

import pickle
from pathlib import Path

from src.collect.gamboo_racecard import Entry
from src.features.assembler import build_features
from src.model.training_data import PL_FEATURES
from src.model.plackett_luce import all_trifecta_probs
from src.model.train_pl import PLModel

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "pl_model.pkl"
DEFAULT_ELO_STATE_PATH = DEFAULT_MODEL_PATH.parent / "elo_state.json"


def save_elo_state(state: dict, path: str | Path = DEFAULT_ELO_STATE_PATH) -> Path:
    """最終Elo {氏名: Elo} をJSONで保存（ライブ予測で選手の現在Eloを引く）。"""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return path


def load_elo_state(path: str | Path = DEFAULT_ELO_STATE_PATH) -> dict:
    """保存済みElo状態を読む。無ければ {}（全員デフォルトElo扱い）。"""
    import json
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_model(model, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """PL線形 / LightGBM(lambdarank) いずれのモデルも保存する（kindで判別）。

    両者とも .strengths(X, car_numbers) を実装し推論経路(strengths_from_model)は共通。
    LightGBMは booster を文字列化して可搬・pickle安全に保存する。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if type(model).__name__ == "GBDTModel":       # lightgbm を必須化しないため型名で判定
        payload = {"kind": "gbdt", "booster_str": model.booster.model_to_string(),
                   "mean": model.mean, "std": model.std,
                   "feature_names": model.feature_names,
                   "standardize_x": getattr(model, "standardize_x", True)}
    else:
        payload = {"kind": "pl", "weights": model.weights, "mean": model.mean,
                   "std": model.std, "feature_names": model.feature_names, "features": PL_FEATURES}
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """保存済みモデルを読む。kind=="gbdt" は GBDTModel、それ以外は PLModel（後方互換）。"""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if d.get("kind") == "gbdt":
        import lightgbm as lgb
        from src.model.train_gbdt import GBDTModel
        booster = lgb.Booster(model_str=d["booster_str"])
        return GBDTModel(booster=booster, mean=d["mean"], std=d["std"],
                         feature_names=d["feature_names"],
                         standardize_x=d.get("standardize_x", True))
    return PLModel(weights=d["weights"], mean=d["mean"], std=d["std"],
                   feature_names=d["feature_names"])


# これが欠けたら平均補完せず推論を諦める特徴（強さの主信号）。周辺特徴（gear_ratio 等）の
# 欠損は補完してよいが、競走得点が無い選手を「平均的な選手」として扱うのは実力を偽装する。
CORE_FEATURES = ("racing_score",)


def strengths_from_model(model: PLModel, entries: list[Entry],
                         recent: dict | None = None,
                         elo_state: dict | None = None,
                         tactics_ctx: dict | None = None,
                         narabi_ctx: dict | None = None,
                         venue_code: str | None = None) -> dict[int, float]:
    """出走選手 → {車番: 1着確率}(Σ=1)。特徴量を組み立てて学習済みモデルで推論する。

    モデルの学習特徴（model.feature_names）に追従。拡張モデルは直近4ヶ月(recent)を、
    Elo付きモデルは elo_state({氏名: Elo}) を、展開特徴付きモデルは tactics_ctx（current_tactics
    の氏名別 as-of history）を必要とする。特徴量が揃わなければ {} を返す。

    venue_code はバンク交互作用列（BANK_KEYS）に使う。**渡されなくても列は必ず作る**
    （重み0＝交互作用なし）。列を作らないと model.feature_names に在る列が df に無く
    KeyError で落ちるため。venue を持たない呼び出し元（api/model.py）がある前提の設計。
    """
    import pandas as pd
    feats = model.feature_names or PL_FEATURES
    df = build_features(entries, recent or {})
    if "rel_elo" in feats:                      # Eloモデル: レース内相対Eloを列追加
        from src.model.elo import DEFAULT_ELO
        state = elo_state or {}
        elos = pd.Series({e.car_number: state.get(e.rider_name, DEFAULT_ELO) for e in entries})
        df["rel_elo"] = elos - elos.mean()
    from src.features.tactics_features import TACTIC_NAMES
    from src.features.bank_features import BANK_KEYS, bank_columns
    tac = None
    need_bank = any(n in feats for n in BANK_KEYS)
    if any(n in feats for n in TACTIC_NAMES) or need_bank:   # 展開特徴を推論と同一関数で付与
        from src.features.rider_tactics import tactics_for_entries
        from src.features.tactics_features import tactic_columns
        tac = tactics_for_entries(entries, recent or {}, tactics_ctx or {})
        cols = tactic_columns(list(df.index), tac)          # {car: [A...B]}
        for i, name in enumerate(TACTIC_NAMES):
            df[name] = [cols[c][i] for c in df.index]
    if need_bank:                               # バンク交互作用（venue不明なら重み0で全て0）
        bcols = bank_columns(list(df.index), tac or {}, venue_code)
        for i, name in enumerate(BANK_KEYS):
            df[name] = [bcols[c][i] for c in df.index]
    from src.features.rider_narabi import NARABI_KEYS
    if any(n in feats for n in NARABI_KEYS):    # 並び予想付きモデル: 3列を推論と同一関数で付与
        from src.features.rider_narabi import narabi_from_order, narabi_columns
        nb = narabi_ctx or {}
        per_car = narabi_from_order(nb.get("order") or [], nb.get("legs") or {})
        ncols = narabi_columns(list(df.index), per_car)
        for i, name in enumerate(NARABI_KEYS):
            df[name] = [ncols[c][i] for c in df.index]
    from src.features.line_features import LINE_KEYS
    if any(n in feats for n in LINE_KEYS):      # 男子モデル: ライン特徴（学習と同一関数）
        from src.features.line_features import line_columns, class_level
        nb = narabi_ctx or {}
        # parse_narabi の lines（ライン境界つき）から {車番: (line_id, pos_in_line)} を作る。
        # lines が無い（＝並び予想を取れなかった）場合は全列0になり、推論は落ちない。
        from src.features.line_features import positions_with_seri, SERI_KEYS
        # 競りの選手は同じライン内位置を共有する（学習と同一関数）。ただし
        # **モデルが ln_seri を持つ時だけ**掛ける。持たないモデル(46列)は競りを直列と
        # みなして学習しているので、ここだけ補正すると train/inference skew になる。
        _seri = (nb.get("seri") or []) if any(n in feats for n in SERI_KEYS) else []
        line_of = {}
        for li, line in enumerate(nb.get("lines") or []):
            gs = [g for g in _seri if all(c in line for c in g)]
            for car, pi in positions_with_seri(line, gs).items():
                line_of[car] = (li, pi)
        scores = {e.car_number: e.racing_score for e in entries if e.racing_score}
        lv = class_level([e.class_rank for e in entries if e.class_rank])
        lcols = line_columns(list(df.index), line_of, scores, lv)
        for i, name in enumerate(LINE_KEYS):
            df[name] = [lcols[c][i] for c in df.index]
    from src.features.line_features import LEGOH_KEYS
    if any(n in feats for n in LEGOH_KEYS):     # 男子モデル: 脚質one-hot（学習と同一関数）
        from src.features.line_features import legoh_columns
        nb = narabi_ctx or {}
        gcols = legoh_columns(list(df.index), nb.get("legs") or {})
        for i, name in enumerate(LEGOH_KEYS):
            df[name] = [gcols[c][i] for c in df.index]
    from src.features.line_features import SERI_KEYS
    if any(n in feats for n in SERI_KEYS):      # 競り参加フラグ（学習と同一関数）
        from src.features.line_features import seri_columns
        nb = narabi_ctx or {}
        scols = seri_columns(list(df.index), nb.get("seri"))
        for i, name in enumerate(SERI_KEYS):
            df[name] = [scols[c][i] for c in df.index]
    # 欠損はレース内平均で補完する。1名の gear_ratio 欠け等だけで全車の推論を捨てて
    # 競走得点ベースラインへ落ちるのを防ぐ（2026-07-28 実測: 本日8R中2Rが該当）。
    # ただし CORE_FEATURES（強さの主信号）が欠けた選手がいる場合と、列が全車欠損の場合は
    # 従来どおり推論を諦める。racing_score 欠損は rel_score_max/score_rank にも波及し
    # 3列がNaNになる＝その選手の実力が全く分からないため、平均で埋めるのは危険。
    sub = df[feats]
    core_missing = [c for c in CORE_FEATURES if c in feats and sub[c].isna().any()]
    if core_missing or sub.isna().all().any():
        return {}
    if sub.isna().any().any():
        df = df.copy()
        df[feats] = sub.fillna(sub.mean())
    cars = list(df.index)
    X = df.loc[cars, feats].to_numpy(dtype=float)
    return model.strengths(X, cars)


def trifecta_from_model(model: PLModel, entries: list[Entry],
                        recent: dict | None = None, elo_state: dict | None = None,
                        venue_code: str | None = None) -> dict[tuple, float]:
    """出走選手 → 三連単210通り確率 {(a,b,c): p}。強さが出せなければ {}。"""
    strengths = strengths_from_model(model, entries, recent, elo_state, venue_code=venue_code)
    return all_trifecta_probs(strengths) if strengths else {}
