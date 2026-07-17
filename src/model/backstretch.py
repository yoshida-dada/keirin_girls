"""展開AI（Stage1）: 最終バック先頭(主導権 B)を取る選手を予測するモデル。

着順モデルと同じ36特徴を入力に、lambdarank で「sb='B'の選手」をtop相当に学習する
（train_gbdt を order=[B取得車] で流用）。出力 strengths(X,cars) は softmax = P(B)。
着順モデルとは別ファイル `data/models/backstretch_model.pkl` に保存。推論は着順モデルと
同一の persist.strengths_from_model で呼べる（feature_names が同じため）。

検証(validate_backstretch_stage): B先頭を55.2%的中（b_count argmax 51.3%/記者予想22.1%を上回る）。
着順への特徴追加は純増なし（主導権は既存特徴に吸収済み）＝本モデルは"推定主導権"の表示用。
"""
from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

from src.model.persist import load_model

BACKSTRETCH_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "backstretch_model.pkl"

_cache: dict = {}


def b_taker(db_path: str | Path) -> dict[str, int]:
    """{race_id: 最終バック先頭の車番}。sb に 'B' を含む選手が一意なレースのみ。"""
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    byrace: dict[str, list[int]] = {}
    for rid, car, v in c.execute("SELECT race_id,car_number,sb FROM results"):
        if v and "B" in v:
            byrace.setdefault(rid, []).append(car)
    c.close()
    return {rid: cs[0] for rid, cs in byrace.items() if len(cs) == 1}


def as_border(samples: list, btk: dict[str, int]) -> list:
    """Stage1学習用: order を [B取得車] に差し替えたサンプル（B一意のレースのみ）。"""
    out = []
    for s in samples:
        b = btk.get(s.race_id)
        if b is None or b not in s.car_numbers:
            continue
        s2 = copy.copy(s)
        s2.order = [b]
        out.append(s2)
    return out


def load_backstretch():
    """本番の展開AIモデルを読む（無ければ None）。プロセス内キャッシュ。"""
    if "m" not in _cache:
        try:
            _cache["m"] = load_model(BACKSTRETCH_PATH)
        except Exception:
            _cache["m"] = None
    return _cache["m"]
