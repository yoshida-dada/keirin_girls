"""S4 EVエンジン (Phase3)。競馬予想プロジェクトの betting/analyzer から移植。

オッズ逆算確率(implied_win_probs) → haircut適用 → logit blend(blend_loglinear) →
買い目選定(build_ev_table 相当) → quarter Kelly(kelly_stake_yen)。
移植元: ../../競馬予想/{analyzer/market.py, betting/ev_table.py, betting/bankroll.py, betting/staking.py}
"""
