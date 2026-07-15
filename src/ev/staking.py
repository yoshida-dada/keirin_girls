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


# ---------------------------------------------------------------------------
# 1レース予算の相対配分（移植元: ../../競馬予想/betting/staking.py: allocate_stakes）。
# quarter Kelly が「資金曲線に対する絶対サイジング」なのに対し、こちらは固定予算
# （1レース予算=BET_HARD_LIMITS["max_stake_per_race_yen"]）を +EV 買い目に按分する。
# 両者は用途が異なる: 実運用は kelly_stake_yen→apply_hard_limits、
# 固定予算のペーパー検証は allocate_stakes、と使い分ける。
# ---------------------------------------------------------------------------

def _weight(row, method: str) -> float:
    """配分の重み。row は EVRow 互換（.ev_net / .prob / .odds を持つ）。"""
    if method == "equal":
        return 1.0
    if method == "ev_prop":
        return max(0.0, row.ev_net)
    if method == "ev_sq":
        return max(0.0, row.ev_net) ** 2
    if method == "kelly":
        if row.odds and row.odds > 1.0:
            edge = row.prob * row.odds - 1.0
            return max(0.0, edge / (row.odds - 1.0))
        return 0.0
    raise ValueError(f"unknown method: {method}")


def allocate_stakes(
    rows: list,
    budget_yen: float = None,
    method: str = "kelly",
    unit: int = BET_UNIT_YEN,
    max_points: int = None,
) -> list[tuple]:
    """+EV 買い目 rows（EVRow、EV降順を想定）に 1レース予算を配分し [(row, stake_yen), ...] を返す。

    - 上位 max_points 点までを対象（既定 BET_HARD_LIMITS["max_points_per_race"]）。
    - method の重み（"equal"/"ev_prop"/"ev_sq"/"kelly"）で按分し unit(100円)へ切り捨て。
    - 端数で余った予算は EV 上位から unit ずつ足して使い切る。
    均等買いではなく期待値・ケリー基準で寄せる（三連単でも EVRow の prob/odds/ev_net をそのまま使う）。
    """
    if budget_yen is None:
        budget_yen = BET_HARD_LIMITS["max_stake_per_race_yen"]
    if max_points is None:
        max_points = BET_HARD_LIMITS["max_points_per_race"]

    rows = list(rows)[:max_points]
    pairs = [(r, w) for r, w in ((r, _weight(r, method)) for r in rows) if w > 0]
    if not pairs:
        return []
    tot = sum(w for _, w in pairs)

    alloc = {}
    for r, w in pairs:
        stake = int((budget_yen * w / tot) // unit) * unit
        if stake >= unit:
            alloc[id(r)] = [r, stake]

    if not alloc:
        # 全て unit 未満になった場合、EV最上位に1点だけ最小単位を置く
        top = pairs[0][0]
        if budget_yen >= unit:
            alloc[id(top)] = [top, unit]

    # 端数の追加配分（EVが高い順に unit を足して予算を使い切る）
    used = sum(v[1] for v in alloc.values())
    leftover = int(budget_yen) - used
    if leftover >= unit:
        for r, _ in pairs:      # pairs はEV降順
            if leftover < unit:
                break
            if id(r) in alloc:
                alloc[id(r)][1] += unit
                leftover -= unit

    return [(v[0], v[1]) for v in alloc.values()]


if __name__ == "__main__":
    # 自己テスト: quarter Kelly / apply_hard_limits / allocate_stakes の最小動作確認。
    from dataclasses import dataclass as _dc

    @_dc
    class _Row:
        combo: tuple
        prob: float
        odds: float
        ev_net: float

    # 1) quarter Kelly: p=0.2, odds=8 → f*=(1.6-1)/7=0.0857, ×0.25=0.0214, ×bankroll 100k ≈ 2142円
    f = kelly_fraction_of_bankroll(0.2, 8.0)
    yen = kelly_stake_yen(100_000, 0.2, 8.0)
    print(f"[kelly] f_frac={f:.4f}  stake(100k)={yen:.0f}円")
    assert abs(f - 0.0857 * 0.25) < 1e-4 and 2100 < yen < 2200

    # 2) apply_hard_limits: 1点上限3000でクリップ＋100円丸め
    st = BankrollState()
    clipped, reason = apply_hard_limits(4200.0, "R1", st)
    print(f"[limits] {clipped}円 / 理由: {reason}")
    assert clipped == 3000 and "1点上限" in reason

    # 3) allocate_stakes: 予算3000を kelly 重みで按分（合計が予算以下・unit倍数）
    rows = [_Row((1, 2, 3), 0.10, 20.0, 1.0),
            _Row((1, 3, 2), 0.05, 30.0, 0.5),
            _Row((2, 1, 3), 0.03, 40.0, 0.2)]
    alloc = allocate_stakes(rows, budget_yen=3000, method="kelly")
    total = sum(s for _, s in alloc)
    print(f"[allocate] {[(r.combo, s) for r, s in alloc]}  合計={total}円")
    assert total <= 3000 and all(s % BET_UNIT_YEN == 0 for _, s in alloc)
    print("OK: staking self-test passed")
