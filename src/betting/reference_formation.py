"""参考フォーメーション（◎頭固定・補正確率top-K）。**実弾非推奨（回収率<100%）**。

回収率100%超のゾーンは存在しない（analyze_upset_odds/validate_himo_roi で実証・控除率25%の壁）。
本モジュールは「黒字買い目」ではなく、精度改善した補正紐で組んだ◎頭固定を、過去実測の的中率/
回収率つきで“参考”提示するためのもの。妙味ポケット(◎が市場に過小評価: 予想先頭/500m/地元◎)を
明示する。買い目選定は corrected_trifecta_probs（本番と同一）で行う。

POCKET_STATS: validate_himo_roi.py(out-of-sample 3432R, walk-forward)の実測 (的中率%, 回収率%)。
"""
from __future__ import annotations

from collections import defaultdict

from src.model.himo_adjust import corrected_trifecta_probs
from src.model.upset import threshold_for
from src.features import venue_meta as vm

# 過去実測（補正選定・◎頭固定top-K）。(hit%, roi%)。回収率は全て<100%＝実弾非推奨。
POCKET_STATS = {
    "全体":      {4: (36.5, 74.3), 6: (45.0, 77.4), 8: (51.0, 82.7)},
    "予想先頭◎": {4: (30.6, 69.1), 6: (39.2, 79.2), 8: (45.2, 79.1)},
    "500mバンク": {4: (38.0, 67.7), 6: (49.3, 80.7), 8: (53.3, 74.2)},
    "地元◎":     {4: (43.6, 89.8), 6: (50.6, 79.0), 8: (57.3, 78.2)},
}
# ポケット優先順（複数該当時の代表）。回収率が相対的に高い順。
_PRIORITY = ["地元◎", "500mバンク", "予想先頭◎"]
DEFAULT_POINTS = 6


def race_pockets(fav: int, narabi_pos: dict | None, venue_code: str, home_of_fav: bool) -> list[str]:
    """このレースの◎が該当する妙味ポケット（市場過小評価）を返す。無ければ空。"""
    out = []
    if home_of_fav:
        out.append("地元◎")
    if vm.bank_length(venue_code) == 500:
        out.append("500mバンク")
    if narabi_pos and narabi_pos.get(fav) == 0:      # ◎が予想先頭
        out.append("予想先頭◎")
    return out


def _combo_str(t) -> str:
    return f"{t[0]}-{t[1]}-{t[2]}"


def build_reference(strengths: dict, narabi_pos: dict | None, venue_code: str,
                    home_of_fav: bool, points: int | None = None,
                    field_size: int = 7) -> dict | None:
    """ガールズ買い目構築: ◎頭固定・補正確率top-K＋三連複2点（実弾非推奨・回収率<100%）。

    点数は**万車券率ベース**（固い<20%=6点で絞る / それ以外=8点で広げる, 1ヶ月検証で現行+5.5pt）。
    三連複は「3人合っても着順違い」を拾う保険（ROIほぼ中立で的中頻度+8pt）。
    """
    if not strengths:
        return None
    fav = max(strengths, key=strengths.get)
    dist = corrected_trifecta_probs(strengths, narabi_pos)
    # 万車券率で点数を決める（固い=絞る/荒れ=広げる）。呼び出しで points 指定時はそれを優先。
    thr = threshold_for(True, field_size)
    upset = sum(p for p in dist.values() if p <= thr) if thr else None
    if points is None:
        points = 6 if (upset is not None and upset < 0.20) else 8
    head = sorted(((o, p) for o, p in dist.items() if o[0] == fav), key=lambda op: -op[1])
    combos = [_combo_str(o) for o, _ in head[:points]]
    hit = round(sum(p for _, p in head[:points]) * 100, 1)     # モデル基準の想定的中率(参考)
    # 三連複 上位2点（順不同3人）
    trp: dict = defaultdict(float)
    for (a, b, c), p in dist.items():
        trp[frozenset((a, b, c))] += p
    trio = ["=".join(str(x) for x in sorted(t))
            for t, _ in sorted(trp.items(), key=lambda kv: -kv[1])[:2]]
    pockets = race_pockets(fav, narabi_pos, venue_code, home_of_fav)
    primary = next((p for p in _PRIORITY if p in pockets), "全体")
    stats = POCKET_STATS.get(primary, POCKET_STATS["全体"])
    return {
        "fav": fav,
        "points": points,
        "combos": combos,
        "trio": trio,                        # 三連複2点(保険)
        "upset": round(upset, 3) if upset is not None else None,
        "band": ("固い<20%" if (upset is not None and upset < 0.20) else "荒れ・標準"),
        "model_hit_pct": hit,               # モデル確率合計（このレース固有の想定的中率）
        "pockets": pockets,                  # 該当妙味ポケット（空=ポケット外）
        "primary": primary,
        "hist": {str(k): {"hit": h, "roi": r} for k, (h, r) in stats.items()},  # 過去実測(pocket平均)
        "note": "参考・実弾非推奨（回収率<100%）。点数は万車券率ベース(固い6/荒れ8)、三連複2点は的中保険。",
    }
