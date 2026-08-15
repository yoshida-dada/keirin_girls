"""波乱確率＝万車券率（三連単の払戻が10,000円以上になる確率）。

**旧表示の何が問題だったか**: ダッシュボードは `1 - 本命の1着確率` を「波乱確率」と出していた。
これは「◎が1着を外す確率」であって高配当になる確率ではない。本命が飛んでも
2番人気→3番人気で決まれば配当は安い。男子25,335Rの実測と突き合わせると
**平均予測57.98% に対し万車券率の実測は28.92%**（約2倍）だった。

**採用した推定量**: 三連単分布のうち `p <= しきい値` の目の確率を合計する。
しきい値は「予測平均が実測の万車券率に一致する値」を過去データから推定する
（`scripts/deploy_upset.py`）。素朴には 控除率25%より 配当 ≈ 0.75/p なので
p <= 0.0075 だが、**実測から推定すると男子7車で 0.0068 前後**とずれる。

**車立てごとに別のしきい値を使う。** 組合せ数が 210(7車) と 504(9車) で倍以上違い、
同じしきいを当てると9車で多くの目が万車券圏に入る。実際、固定しきい 0.0075 では
9車が実測45.4%に対し65.8%（+20pt）と大きく過大だった。9車は男子の9.3%しかなく、
全体プールのECE(0.026)には埋もれていた。

**検証（`scripts/validate_upset_prob.py`, walk-forward, 車立て別を主基準）**:
  男子7車     予測26.8% / 実測27.0% / ECE 0.0148 / Brier 0.1905(定数0.1970) / ρ0.99 → 採用
  ガールズ7車 予測16.7% / 実測17.1% / ECE 0.0211 / Brier 0.1334(定数0.1420) / ρ0.93 → 採用
  男子9車     予測48.0% / 実測45.1% / ECE 0.0523 → **不採用**。
              n=1,130（十分位113件）では実測率45%の標準誤差が約4.7ptあり、
              完全に較正されたモデルでもECEは0.038前後になる＝0.03は到達不能な基準
              だった。基準は動かさず不採用とし、**この層には万車券率を出さない**。

モデル分布×実オッズ（市場が100倍以上に値付けした目の確率を合計）も試したが
平均50.33%と大きく過大で不採用。モデルは市場より裾に確率を置きすぎている。
"""
from __future__ import annotations

import json
from pathlib import Path

MAN_YEN = 10000            # 万車券のしきい（100円あたりの払戻）
TAKEOUT = 0.75             # 控除率25%。配当 ≈ TAKEOUT / p
P_MAN = TAKEOUT / (MAN_YEN / 100)          # 0.0075。実測推定が無いときの既定値

THRESHOLD_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "upset_thresholds.json"

_cache: dict | None = None


def _thresholds() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(Path(THRESHOLD_PATH).read_text(encoding="utf-8"))
        except Exception:
            _cache = {}
    return _cache


def threshold_for(is_girls: bool, field_size: int | None) -> float | None:
    """その層のしきい値。**検証を通していない層は None**（＝万車券率を出さない）。"""
    if field_size is None:
        return None
    t = _thresholds().get("girls" if is_girls else "men", {}).get(str(int(field_size)))
    return float(t) if t is not None else None


def man_prob(probs: dict[tuple, float] | None, is_girls: bool = False,
             field_size: int | None = None) -> float | None:
    """三連単の全通り確率 → 万車券率。検証を通していない層では None を返す。

    probs は {(1着,2着,3着): p}（`all_trifecta_probs` の出力。合計1）。
    None を返す場合に既定値で代用しない。**出せない層に数字を出さない**のが要点で、
    男子9車に固定しきいを当てると実測45%に対し66%と出てしまう。
    """
    if not probs:
        return None
    t = threshold_for(is_girls, field_size if field_size is not None else _infer_size(probs))
    if t is None:
        return None
    return round(sum(p for p in probs.values() if p <= t), 4)


def _infer_size(probs: dict[tuple, float]) -> int | None:
    """出走数を目から復元する（field_size を渡し忘れた呼び出しの保険）。"""
    cars = {c for k in probs for c in k}
    return len(cars) or None
