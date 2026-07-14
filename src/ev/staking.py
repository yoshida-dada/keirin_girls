"""資金管理（S4。移植元: ../../競馬予想/betting/bankroll.py, 課題F）。

quarter Kelly で賭け金を決め、ハードリミット（1点/1レース/1日の上限・最大点数・連敗kill switch）で
クリップする。実際の購入は行わず金額計算のみ。まずペーパートレードで記録する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import KELLY_FRACTION, BET_HARD_LIMITS, BET_UNIT_YEN


def round_to_unit(stake_yen: float, unit: int = BET_UNIT_YEN) -> int:
    """賭け金を最小単位（既定100円）の倍数に切り捨てる。単位未満は0。"""
    if stake_yen < unit:
        return 0
    return int(stake_yen // unit) * unit


def kelly_fraction_of_bankroll(
    prob: float, odds: float, kelly_fraction: float = KELLY_FRACTION
) -> float:
    """フラクショナル・ケリー: f* = (p×odds − 1)/(odds − 1) に kelly_fraction を掛けた
    「資金に対する賭け金割合」（0〜1にクリップ）。EVが無い/オッズ不正なら0.0。"""
    if odds is None or odds <= 1.0 or prob <= 0.0:
        return 0.0
    edge = prob * odds - 1.0
    if edge <= 0.0:
        return 0.0
    f_star = edge / (odds - 1.0)
    return max(0.0, min(1.0, f_star * kelly_fraction))


def kelly_stake_yen(
    bankroll_yen: float, prob: float, odds: float, kelly_fraction: float = KELLY_FRACTION
) -> float:
    """kelly_fraction_of_bankroll を円額に変換（クリップ前の理論値）。"""
    return bankroll_yen * kelly_fraction_of_bankroll(prob, odds, kelly_fraction)


@dataclass
class BankrollState:
    """当日の購入状況（ハードリミット判定用の可変状態）。"""
    day_total_stake_yen: float = 0.0
    race_totals_yen: dict = field(default_factory=dict)
    race_points: dict = field(default_factory=dict)   # レース別の購入点数
    consecutive_losses: int = 0
    kill_switch_triggered: bool = False

    def record_bet(self, race_id: str, stake_yen: float) -> None:
        self.day_total_stake_yen += stake_yen
        self.race_totals_yen[race_id] = self.race_totals_yen.get(race_id, 0.0) + stake_yen
        self.race_points[race_id] = self.race_points.get(race_id, 0) + 1

    def record_result(self, won: bool) -> None:
        """レース確定後に呼ぶ。連敗が続くとkill switchが立つ。"""
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1


def apply_hard_limits(
    stake_yen: float,
    race_id: str,
    state: BankrollState,
    limits: dict = None,
) -> tuple[float, str]:
    """理論上の賭け金を 1点/1レース/1日の上限・最大点数・kill switch でクリップし
    最小単位(100円)へ丸める。戻り値: (丸め後賭け金, 理由。問題なしなら空文字列)。"""
    limits = limits if limits is not None else BET_HARD_LIMITS
    if state.kill_switch_triggered or state.consecutive_losses >= limits["max_consecutive_losses"]:
        state.kill_switch_triggered = True
        return 0.0, f"kill switch発動（連敗{state.consecutive_losses}回）につき購入停止"

    max_pts = limits.get("max_points_per_race")
    if max_pts is not None and state.race_points.get(race_id, 0) >= max_pts:
        return 0.0, f"最大購入点数({max_pts}点)に到達につき購入対象外"

    stake = max(0.0, stake_yen)
    reasons = []
    if stake > limits["max_stake_per_bet_yen"]:
        stake = limits["max_stake_per_bet_yen"]
        reasons.append("1点上限")
    race_room = max(0.0, limits["max_stake_per_race_yen"] - state.race_totals_yen.get(race_id, 0.0))
    if stake > race_room:
        stake = race_room
        reasons.append("1レース予算")
    day_room = max(0.0, limits["max_stake_per_day_yen"] - state.day_total_stake_yen)
    if stake > day_room:
        stake = day_room
        reasons.append("1日上限")

    stake = round_to_unit(stake)
    return stake, ("・".join(reasons) + "でクリップ" if reasons else "")
