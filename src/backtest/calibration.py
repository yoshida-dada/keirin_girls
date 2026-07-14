"""キャリブレーション検証（課題B / Phase4）。

モデル確率が実現頻度と一致しているかを reliability curve・Brier score で測る。ROIが良く見えても
キャリブレーションが崩れているバケット（レースタイプ×オッズ帯）は運用から除外する（仕様書課題B）。

三連単の買い目ごとに「モデル確率 p」と「的中したか（0/1）」の組を集計する。学習器の比較
（PL線形 vs LightGBM）にも同じ指標を使う。
"""
from __future__ import annotations

from dataclasses import dataclass


def brier_score(pairs: list[tuple[float, int]]) -> float | None:
    """Brier score = mean((p − y)^2)。小さいほど良い。pairs=[(確率, 的中0/1)]。空ならNone。"""
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    mean_pred: float | None     # そのビンの平均予測確率
    emp_freq: float | None      # そのビンの実現的中率
    count: int


def reliability_curve(pairs: list[tuple[float, int]], n_bins: int = 10
                      ) -> list[ReliabilityBin]:
    """予測確率を [0,1] の n_bins 等幅ビンに分け、各ビンの平均予測と実現頻度を返す。

    よく較正されていれば mean_pred ≈ emp_freq（対角線に乗る）。
    """
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in pairs:
        idx = min(int(p * n_bins), n_bins - 1) if p >= 0 else 0
        bins[idx].append((p, y))
    out: list[ReliabilityBin] = []
    for i, b in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if b:
            mean_pred = sum(p for p, _ in b) / len(b)
            emp = sum(y for _, y in b) / len(b)
        else:
            mean_pred = emp = None
        out.append(ReliabilityBin(lo=lo, hi=hi, mean_pred=mean_pred, emp_freq=emp, count=len(b)))
    return out


def expected_calibration_error(pairs: list[tuple[float, int]], n_bins: int = 10) -> float | None:
    """ECE = Σ (ビン件数/全体) × |平均予測 − 実現頻度|。0が完全較正。"""
    if not pairs:
        return None
    total = len(pairs)
    ece = 0.0
    for b in reliability_curve(pairs, n_bins):
        if b.count and b.mean_pred is not None:
            ece += (b.count / total) * abs(b.mean_pred - b.emp_freq)
    return ece


def calibration_by_bucket(
    records: list[tuple[str, float, int]], n_bins: int = 10
) -> dict[str, dict]:
    """バケット別のキャリブレーション指標を返す。records=[(bucket, 確率, 的中0/1)]。

    戻り値: {bucket: {"n", "brier", "ece", "mean_pred", "emp_freq"}}。
    ROIが良くても ece/brier が悪いバケットは運用除外の判断材料にする（課題B）。
    """
    buckets: dict[str, list[tuple[float, int]]] = {}
    for bucket, p, y in records:
        buckets.setdefault(bucket, []).append((p, y))
    out: dict[str, dict] = {}
    for bucket, pairs in buckets.items():
        n = len(pairs)
        out[bucket] = {
            "n": n,
            "brier": brier_score(pairs),
            "ece": expected_calibration_error(pairs, n_bins),
            "mean_pred": sum(p for p, _ in pairs) / n if n else None,
            "emp_freq": sum(y for _, y in pairs) / n if n else None,
        }
    return out
