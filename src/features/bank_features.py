"""バンク長との交互作用列を学習・推論で**共通生成**する（train/inference skew防止）。

なぜ交互作用でしか入れられないか:
  バンク長はレース定数で、順位モデル（PL/lambdarank）はシフト不変。race一定の列は
  設計上そのまま入れても効かないため除外されている（training_data の方針）。
  一方で実測ではバンクが決まり手を大きく変える:
      逃げ率 333m 35.3% / 400m 19.3% / 500m 12.8%（全体21.4%）
      ペースで層別しても消えない（スロー内で 333m 38.6% vs 500m 13.4%）
  そこで「レース内で変動する量 × バンクの逃げ有利度」の形にすると、レース内で値が
  ばらつくためモデルが利用できる。これが唯一バンク情報を注入できる形。

列:
  x_esc_bank  = 逃げ残率(レース内相対) × bank_w   … 短走路ほど逃げ残りが効く
  x_lead_bank = 主導権指数(レース内相対) × bank_w … 短走路ほど主導権が効く

bank_w は `kimarite_stats.json`（scripts/analyze_gear_bank.py --emit が生成）の
バンク別逃げ率と全体との差を1/10スケールにしたもの（333m +1.39 / 400m −0.21 / 500m −0.86）。
バンク不明は 0（＝交互作用なし＝従来と同じ挙動）にフォールバックする。

walk-forward検証（scripts/validate_bank_interaction.py）:
  5fold  全体tri10 +0.92pt(4/5) top1 +0.49pt(4/5) 333m top1 +0.99pt(4/5)
  7fold  全体tri10 +0.66pt(5/7) top1 +0.43pt(4/7) 333m top1 +0.76pt(3/7)
  主基準（tri10が過半で改善・top1が悪化しない）は両条件で再現。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.features import venue_meta as vm
from src.features.tactics_features import _rel

# モデル入力の列名（この順で連結する）
BANK_KEYS = ["x_esc_bank", "x_lead_bank"]
# 交互作用の元になるレース内相対量（rider_tactics の生キー）
_SRC_KEYS = ["escape_survival", "lead_index"]

STATS_PATH = Path(__file__).resolve().parent.parent / "model" / "kimarite_stats.json"


@lru_cache(maxsize=1)
def _weights() -> dict[int, float]:
    """バンク長 → 逃げ有利度。統計が無ければ空（＝全て0扱い）。"""
    try:
        d = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    g = ((d.get("global") or {}).get("kim") or {}).get("逃")
    if g is None:
        return {}
    out = {}
    for b, cell in (d.get("bank") or {}).items():
        k = (cell or {}).get("kim") or {}
        if "逃" in k:
            out[int(b)] = (k["逃"] - g) / 10.0
    return out


def bank_weight(venue_code: str | None) -> float:
    """開催場コード → バンクの逃げ有利度。不明は0（交互作用を効かせない）。"""
    bank = vm.bank_length(venue_code or "") if venue_code else None
    return _weights().get(bank, 0.0) if bank else 0.0


def bank_columns(cars: list[int], tac_by_car: dict[int, dict],
                 venue_code: str | None) -> dict[int, list]:
    """出走車と raw 展開特徴から交互作用列を車番キーで返す（BANK_KEYS 順）。

    相対化は tactics_features._rel と同一関数を使う＝t_escape_rel / t_lead_rel と
    同じ値に bank_w を掛ける形になり、学習・推論で必ず一致する。
    """
    w = bank_weight(venue_code)
    cols = {c: [] for c in cars}
    for key in _SRC_KEYS:
        rel = _rel([tac_by_car.get(c, {}).get(key) for c in cars])
        for c, v in zip(cars, rel):
            cols[c].append(v * w)
    return cols
