"""(c) 男子の展開分岐: 「誰が主導権を取るか」で場合分けし、分岐ごとに買い目を出す。

出力イメージ:
    展開A  5番ライン主導権 62%  → 5-21-2137
    展開B  3番ライン主導権 21%  → 3-74-1745
    展開C  単騎6番が主導権  9%  → 6-53-1352

**作り方**（手で係数を決めない。決めるとLEG_AGGRの二の舞になる）:
  1. 主導権の確率 P(B=b) は展開AI（男子62.5%的中・記者先頭49.4%/B回数最大56.3%を上回る）
  2. 各分岐の着順分布は s'_c = s_c × w[役割(c | B=b)] で作る。
     w は `branch_stats_men.json` の実測倍率＝「実測の役割別シェア ÷ 素のPLの役割別シェア」。
     **モデルの実力評価はそのまま残し、役割による偏りだけを実測に合わせる**。
  3. 分岐の三連単分布から買い目（フォーメーション）を組む。

役割は主導権者 b から見た相対位置: B本人 / B番手 / B同ライン他 / 他ライン先頭 /
他ライン番手 / 他ライン3番手+ / 単騎。実測(25,238R)では B本人が1着31.2%・B番手28.4%で、
**1着の約6割が主導権ラインから出る**。

**この分岐は買い目の推奨ではない。** 期待値ゾーンは検証済みで存在しない（Phase5: 全点均等買い
ROI 58.0%・上限75%）。展開の読みと紐選びの材料として出す。
"""
from __future__ import annotations

import json
from functools import lru_cache
from itertools import permutations
from pathlib import Path

STATS_PATH = Path(__file__).with_name("branch_stats_men.json")
ROLES = ["B本人", "B番手", "B同ライン他", "他ライン先頭", "他ライン番手", "他ライン3番手+", "単騎"]


@lru_cache(maxsize=1)
def load_stats() -> dict:
    try:
        return json.loads(STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def role_of(car: int, b: int, lines: list[list[int]]) -> str:
    """主導権者 b から見た car の役割。lines は記者の並び予想のライン構成。"""
    line_of = {c: i for i, mem in enumerate(lines) for c in mem}
    lb, lc = line_of.get(b), line_of.get(car)
    if lc is None or lb is None:
        return "単騎"
    if lc == lb:
        if car == b:
            return "B本人"
        mem = lines[lb]
        return "B番手" if mem.index(car) == mem.index(b) + 1 else "B同ライン他"
    mem = lines[lc]
    if len(mem) == 1:
        return "単騎"
    i = mem.index(car)
    return "他ライン先頭" if i == 0 else ("他ライン番手" if i == 1 else "他ライン3番手+")


def mate_of(car: int, lines: list[list[int]]) -> int | None:
    """car の直後を追走する同一ラインの選手（＝番手）。最後尾・単騎は None。"""
    for mem in lines:
        if car in mem:
            i = mem.index(car)
            return mem[i + 1] if i + 1 < len(mem) else None
    return None


def branch_trifecta(strengths: dict[int, float], b: int, lines: list[list[int]],
                    stats: dict | None = None,
                    mate_boost: float | None = None) -> dict[tuple, float]:
    """B=b を条件にした三連単分布。役割倍率＋**1着に連動した番手加点**で重みを付ける。

    役割倍率だけでは足りない。あれは1着・2着の重みを**別々に**掛けるだけなので、
    「同じラインが1着と2着を占める」という**同時性**を作れない。実測(直近2,000R)では
    真の主導権者を与えてもライン決着 37.3% vs 実測 55.1% と17.8pt低いままだった。
    そこで2着の重みを「実際に1着になった車の番手」に対して加点する
    （＝1着が決まってから2着を引くという逐次構造を、依存関係つきで書く）。
    """
    st = stats or load_stats()
    w = st.get("weights") or {}
    if not w:
        from src.model.plackett_luce import all_trifecta_probs
        return all_trifecta_probs(strengths)
    riders = [c for c in strengths if strengths[c] > 0]
    if len(riders) < 3:
        return {}
    role = {c: role_of(c, b, lines) for c in riders}
    wt = {k: {c: strengths[c] * (w.get(k) or {}).get(role[c], 1.0) for c in riders}
          for k in ("1", "2", "3")}
    mb = st.get("mate_boost", 0.0) if mate_boost is None else mate_boost
    out: dict[tuple, float] = {}
    z1 = sum(wt["1"].values())
    for a1 in riders:
        p1 = wt["1"][a1] / z1 if z1 > 0 else 0.0
        if p1 <= 0:
            continue
        rem2 = [c for c in riders if c != a1]
        # 1着 a1 の番手を加点する。ここが「ライン決着」の同時性を作る唯一の場所。
        w2 = dict(wt["2"])
        m = mate_of(a1, lines)
        if mb and m in w2:
            w2[m] *= (1.0 + mb)
        z2 = sum(w2[c] for c in rem2)
        for a2 in rem2:
            p2 = w2[a2] / z2 if z2 > 0 else 0.0
            rem3 = [c for c in rem2 if c != a2]
            z3 = sum(wt["3"][c] for c in rem3)
            if z3 <= 0:
                continue
            base = p1 * p2 / z3
            for a3 in rem3:
                p = base * wt["3"][a3]
                if p > 0:
                    out[(a1, a2, a3)] = p
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s > 0 else out


def _formation(dist: dict[tuple, float], n1: int = 1, n2: int = 3, n3: int = 5) -> dict:
    """分岐の分布 → フォーメーション（1着n1頭 × 2着n2頭 × 3着n3頭）。

    **条件付きで選ぶ**。単純に各着の周辺確率で選ぶと、1着に固定した車が2着欄にも入り
    枠を1つ無駄にする（実際それで3点しか買えないフォーメーションが出た）。
    2着は「1着が f1 のいずれか」を条件に、3着はさらに「2着が f2 のいずれか」を条件に選ぶ。

    cover は**その買い方でこの分岐が当たる確率**。回収率ではない。
    """
    if not dist:
        return {}
    top = lambda d, k: [c for c, _ in sorted(d.items(), key=lambda kv: -kv[1])[:k]]

    p1: dict[int, float] = {}
    for (a, _b, _c), p in dist.items():
        p1[a] = p1.get(a, 0.0) + p
    f1 = top(p1, n1)

    p2: dict[int, float] = {}
    for (a, b, _c), p in dist.items():
        if a in f1:
            p2[b] = p2.get(b, 0.0) + p
    f2 = top(p2, n2)

    p3: dict[int, float] = {}
    for (a, b, c), p in dist.items():
        if a in f1 and b in f2:
            p3[c] = p3.get(c, 0.0) + p
    f3 = top(p3, n3)

    combos = {(a, b, c) for a in f1 for b in f2 for c in f3 if len({a, b, c}) == 3}
    j = lambda xs: "".join(str(x) for x in sorted(xs))
    return {
        "first": f1, "second": f2, "third": f3,
        "points": len(combos),
        "text": f"{j(f1)}-{j(f2)}-{j(f3)}",
        "cover": round(sum(dist.get(c, 0.0) for c in combos), 4),
    }


def build_branches(strengths: dict[int, float], lines: list[list[int]],
                   b_probs: dict[int, float] | None, names: dict[int, str] | None = None,
                   top_k: int = 3, min_prob: float = 0.05) -> dict | None:
    """展開分岐を作る。lines か b_probs が無ければ None（ガールズ・並び予想なし）。

    b_probs は展開AIの P(B)。top_k 個まで、確率 min_prob 以上の分岐を返す。
    """
    st = load_stats()
    if not lines or not b_probs or not strengths or not st:
        return None
    names = names or {}
    line_of = {c: i for i, mem in enumerate(lines) for c in mem}
    cand = sorted(b_probs.items(), key=lambda kv: -kv[1])[:top_k]
    out = []
    for b, pb in cand:
        if pb < min_prob or b not in strengths:
            continue
        dist = branch_trifecta(strengths, b, lines, st)
        if not dist:
            continue
        li = line_of.get(b)
        mem = lines[li] if li is not None else [b]
        solo = len(mem) == 1
        out.append({
            "b_car": b,
            "b_name": names.get(b),
            "prob": round(pb, 4),
            "line": mem,
            "is_solo": solo,
            "label": (f"単騎{b}番が主導権" if solo else f"{mem[0]}番ラインが主導権"),
            # この分岐での各車1着確率（読み用）
            "win": {int(c): round(sum(p for (a, _, _), p in dist.items() if a == c), 4)
                    for c in strengths},
            "formation": _formation(dist),
        })
    if not out:
        return None
    return {
        "branches": out,
        "covered": round(sum(x["prob"] for x in out), 4),
        "hit_rate": 0.625,      # 主導権予測の out-of-sample 実測（記者先頭49.4%）
        "note": "主導権の確率は展開AI（男子62.5%的中）。分岐内の着順は実測の役割別シェアへ"
                "合わせた条件付き分布。買い目は展開の読み・紐選びの材料であって推奨ではない"
                "（期待値ゾーンは検証済みで存在しない: 全点均等買いROI 58.0%・上限75%）。",
    }
