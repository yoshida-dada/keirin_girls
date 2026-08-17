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
# 買い目の点数予算。この範囲でカバー率が最大になるよう枠を貪欲に広げる。
# 「カバー率N%まで広げる」方式は駄目だった: 1着を1頭に固定すると到達可能な上限が
# その車の1着確率(27〜46%)しかなく、届かない目標だと全車に広がる（実測で確認）。
FORMATION_BUDGET = 18
# 全分岐をケアする買い目の総点数（重複除去後の目標）。分岐ごとに満額で買うと
# 和集合で28〜31点になるので、その6割前後に収める。
# **1つの形に押し込まない**。展開Aが「3が頭」・Bが「1が頭」なら1つの長方形では
# 1着に3と1を両方入れる分だけ2着3着を削る羽目になり、かえって効率が落ちる。
# 複数フォーメーションのまま、点数を P(展開) に比例配分して削る。
PLAN_POINTS = 18
PLAN_MIN_PER_BRANCH = 3
# 合成オッズを上げるための足切り。**この倍率以下の目を落とす**。
# 落とした目は必ず一覧で出す（黙って消すと買い目が変わったことに気づけない）。
PLAN_ODDS_FLOOR = 10.0
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


def line_mates_of(car: int, lines: list[list[int]]) -> list[int]:
    """car と同一ラインの**直後以外**の選手（3番手や、carが番手なら先頭）。"""
    for mem in lines:
        if car in mem:
            i = mem.index(car)
            return [x for j, x in enumerate(mem) if j != i and j != i + 1]
    return []


def branch_trifecta(strengths: dict[int, float], b: int, lines: list[list[int]],
                    stats: dict | None = None,
                    mate_boost: float | None = None,
                    line_boost: float | None = None,
                    fav_fade: float | None = None) -> dict[tuple, float]:
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
    lb = st.get("line_boost", 0.0) if line_boost is None else line_boost
    # ◎が1着を外したときの失速。実測では**◎が負けた場合の44.0%が3着圏外**まで沈む
    # （2着34.7% / 3着21.3%）。PLは強い選手を2・3着に残しすぎるので、
    # 1着が◎以外に決まった時点で◎の2着/3着の重みを落とす。
    # これが無いと「◎が飛ぶ」ケースを 15.9%(実測23.2%) と 7.3pt 過小評価する。
    ff = st.get("fav_fade", 1.0) if fav_fade is None else fav_fade
    fav = max(strengths, key=strengths.get) if strengths else None
    out: dict[tuple, float] = {}
    z1 = sum(wt["1"].values())
    for a1 in riders:
        p1 = wt["1"][a1] / z1 if z1 > 0 else 0.0
        if p1 <= 0:
            continue
        rem2 = [c for c in riders if c != a1]
        # 1着 a1 と同じラインの選手を加点する。ここが「ライン決着」の同時性を作る唯一の場所。
        # **番手と、それ以外の同ライン**を別係数にする。合計だけ合わせると配分を誤るため:
        # mate_boost だけで較正したとき、合計は合う(55.6% vs 実測55.3%)のに
        # 番手 42.2%(実測32.2%) / 同ライン他 13.4%(実測23.1%) と ±10pt ずれていた。
        w2 = dict(wt["2"])
        m = mate_of(a1, lines)
        if mb and m in w2:
            w2[m] *= (1.0 + mb)
        if lb:
            for x in line_mates_of(a1, lines):
                if x in w2:
                    w2[x] *= (1.0 + lb)
        w3 = wt["3"]
        if ff != 1.0 and fav is not None and a1 != fav:
            if fav in w2:
                w2 = dict(w2)
                w2[fav] *= ff
            w3 = dict(w3)
            if fav in w3:
                w3[fav] *= ff
        z2 = sum(w2[c] for c in rem2)
        for a2 in rem2:
            p2 = w2[a2] / z2 if z2 > 0 else 0.0
            rem3 = [c for c in rem2 if c != a2]
            z3 = sum(w3[c] for c in rem3)
            if z3 <= 0:
                continue
            base = p1 * p2 / z3
            for a3 in rem3:
                p = base * w3[a3]
                if p > 0:
                    out[(a1, a2, a3)] = p
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s > 0 else out


def _formation(dist: dict[tuple, float], budget: int = 18) -> dict:
    """分岐の分布 → フォーメーション。**点数の予算内でカバー率を最大化する**。

    点数を 1×3×5 のように固定してはいけない。分岐の分布の尖り方はレースごとに違うので、
    固定するとカバーが足りない分岐と無駄に広い分岐が同じ点数になる（実際23%と31%が同じ12点）。
    逆に「カバー率N%まで広げる」も駄目で、1着を1頭に固定すると到達可能な上限がその車の
    1着確率(27〜46%)しかなく、届かない目標を置くと全車に広がってしまう（実測で確認）。

    そこで **点数の予算を決め、1点あたりのカバー増が最大の枠を貪欲に広げる**。
    1着/2着/3着のどれを広げるかもデータに決めさせる（実測の非対称性
    「2着はライン内56%・3着は他ライン63%」は条件付き分布が既に持っているので、
    貪欲法が自然に3着を広く取る）。

    返す指標:
      cover      … その買い方でこの分岐が当たる確率（**回収率ではない**）
      cover_cond … 1着が当たった前提でのカバー率（= cover / P(1着∈f1)）
    """
    if not dist:
        return {}
    top = lambda d, k: [c for c, _ in sorted(d.items(), key=lambda kv: -kv[1])[:k]]

    def build(k1, k2, k3):
        p1: dict[int, float] = {}
        for (a, _b, _c), p in dist.items():
            p1[a] = p1.get(a, 0.0) + p
        f1 = top(p1, k1)
        p2: dict[int, float] = {}
        for (a, b, _c), p in dist.items():
            if a in f1:
                p2[b] = p2.get(b, 0.0) + p
        f2 = top(p2, k2)
        p3: dict[int, float] = {}
        for (a, b, c), p in dist.items():
            if a in f1 and b in f2:
                p3[c] = p3.get(c, 0.0) + p
        f3 = top(p3, k3)
        combos = {(a, b, c) for a in f1 for b in f2 for c in f3 if len({a, b, c}) == 3}
        return f1, f2, f3, combos, sum(dist.get(c, 0.0) for c in combos), sum(p1[c] for c in f1)

    n = len({c for combo in dist for c in combo})
    k = [1, 2, 3]
    f1, f2, f3, combos, cov, p1sum = build(*k)
    while True:
        best = None
        for i in range(3):
            if k[i] >= n:
                continue
            k2 = list(k)
            k2[i] += 1
            *_, cb, cv, _ps = build(*k2)
            add = len(cb) - len(combos)
            if add <= 0 or len(cb) > budget:
                continue
            gain = (cv - cov) / add
            if best is None or gain > best[0]:
                best = (gain, i)
        if best is None:
            break
        k[best[1]] += 1
        f1, f2, f3, combos, cov, p1sum = build(*k)

    j = lambda xs: "".join(str(x) for x in sorted(xs))
    return {
        "first": f1, "second": f2, "third": f3,
        "points": len(combos),
        "text": f"{j(f1)}-{j(f2)}-{j(f3)}",
        "cover": round(cov, 4),
        "cover_cond": round(cov / p1sum, 4) if p1sum > 0 else None,
    }


def synth_odds(form: dict, odds: dict[tuple, float] | None) -> dict | None:
    """買い目の合成オッズ。odds が無ければ None（発売前は出せない）。

    合成オッズ = 1 / Σ(1/オッズ)。**どの目で当たっても同じ額が返るように資金を配分した
    ときの倍率**で、均等買いの平均ではない。プロジェクト内の `synth_odds_1st` と同じ定義。

    均等買いだと当たる目によって回収が何倍も変わるので、レンジ(min/max)も併記する。
    オッズが取れていない目がある場合は covered で何点ぶん取れているかを返す
    （締切直前の更新で全点そろうが、遠いレースは間引いているため）。
    """
    if not form or not odds:
        return None
    combos = [(a, b, c) for a in form.get("first", []) for b in form.get("second", [])
              for c in form.get("third", []) if len({a, b, c}) == 3]
    vals = [odds[k] for k in combos if odds.get(k) and odds[k] > 0]
    if not vals:
        return None
    inv = sum(1.0 / o for o in vals)
    return {"synth": round(1.0 / inv, 1) if inv > 0 else None,
            "min": round(min(vals), 1), "max": round(max(vals), 1),
            "covered": len(vals), "points": len(combos)}


def _combos(f) -> list[tuple]:
    return [(a, b, c) for a in (f or {}).get("first", [])
            for b in (f or {}).get("second", []) for c in (f or {}).get("third", [])
            if len({a, b, c}) == 3]


def build_plan(dists: list, mix: dict, pmodel, odds: dict | None,
               total: int = PLAN_POINTS, floor: float = PLAN_ODDS_FLOOR) -> dict | None:
    """全分岐をケアする買い目。**複数フォーメーションのまま**点数を配分して削る。

    点数は P(展開) に比例配分する（起きやすい展開に厚く張る）。1つの長方形に
    押し込まない理由は、展開ごとに1着候補が違うと1つの形では表現が歪むため。

    足切り: `floor` 倍以下の目を落とす。合成オッズは 1/Σ(1/o) なので安い目を
    落とすほど上がる。**落とした目は必ず一覧で返す**（黙って消すと買い目が
    変わったことに気づけない）。足切り前後の合成オッズを両方返す。

    注意: 足切りで回収率が上がるわけではない（実測では分散の付け替え）。
    ここで返すのは「合成オッズをいくつまで上げられるか」という情報。
    """
    if not dists:
        return None
    z = sum(p for _p, p, _d in [(b, p, d) for b, p, d in dists]) or 1.0
    forms, used = [], set()
    for b, pb, dist in dists:
        bud = max(PLAN_MIN_PER_BRANCH, round(total * pb / z))
        f = _formation(dist, budget=bud)
        if not f:
            continue
        forms.append({"b_car": b, "prob": round(pb, 4), "text": f["text"],
                      "first": f["first"], "second": f["second"], "third": f["third"],
                      "points": f["points"]})
        used |= set(_combos(f))
    if not forms:
        return None

    def stat(cs):
        vals = [odds[k] for k in cs if odds and odds.get(k)] if odds else []
        inv = sum(1.0 / o for o in vals) if vals else 0.0
        return {"points": len(cs),
                "p_model": round(sum(mix.get(k, 0.0) for k in cs), 4),
                "synth": round(1.0 / inv, 1) if inv > 0 else None,
                "min": round(min(vals), 1) if vals else None,
                "max": round(max(vals), 1) if vals else None}

    out = {"forms": forms, "before": stat(used), "floor": floor}
    if odds:
        cut = sorted(k for k in used if odds.get(k) and odds[k] <= floor)
        keep = used - set(cut)
        out["cut"] = [{"combo": f"{a}-{b}-{c}", "odds": odds[(a, b, c)]} for a, b, c in cut]
        out["after"] = stat(keep) if keep else None
    return out


def build_branches(strengths: dict[int, float], lines: list[list[int]],
                   b_probs: dict[int, float] | None, names: dict[int, str] | None = None,
                   top_k: int = 3, min_prob: float = 0.05,
                   odds: dict[tuple, float] | None = None) -> dict | None:
    """展開分岐を作る。lines か b_probs が無ければ None（ガールズ・並び予想なし）。

    b_probs は展開AIの P(B)。top_k 個まで、確率 min_prob 以上の分岐を返す。
    """
    st = load_stats()
    if not lines or not b_probs or not strengths or not st:
        return None
    names = names or {}
    line_of = {c: i for i, mem in enumerate(lines) for c in mem}
    fav = max(strengths, key=strengths.get) if strengths else None   # ◎＝モデル1着確率トップ
    cand = sorted(b_probs.items(), key=lambda kv: -kv[1])[:top_k]
    # 先に全分岐の分布を作り、**混合分布**を用意する。
    # 分岐ごとの cover は「その展開が起きた前提」の確率であって、実際に買ったときの
    # 的中確率ではない（他の展開でも当たり得る）。買い目の点数を買ったら何%当たるかは
    # 混合分布 Σ P(b)·P(順位|b) で合計しないと出ない。
    dists = []
    for b, pb in cand:
        if pb < min_prob or b not in strengths:
            continue
        d = branch_trifecta(strengths, b, lines, st)
        if d:
            dists.append((b, pb, d))
    mix: dict[tuple, float] = {}
    for _b, pb, d in dists:
        for k, v in d.items():
            mix[k] = mix.get(k, 0.0) + pb * v
    _z = sum(mix.values())
    if _z > 0:
        mix = {k: v / _z for k, v in mix.items()}

    def _pmodel(f):
        """その買い目を実際に買ったときの的中確率（混合分布での合計）。"""
        if not f:
            return None
        s = sum(mix.get((a, b2, c), 0.0)
                for a in f.get("first", []) for b2 in f.get("second", [])
                for c in f.get("third", []) if len({a, b2, c}) == 3)
        return round(s, 4)

    out = []
    for b, pb, dist in dists:
        li = line_of.get(b)
        mem = lines[li] if li is not None else [b]
        solo = len(mem) == 1
        _f = _formation(dist, budget=FORMATION_BUDGET)
        _ft = formation_types(dist, fav) if fav is not None else []
        # p_model = その点数を買ったときの的中確率（混合分布での合計）。
        # cover（型内）は展開を前提にした条件付きなので別物。
        if _f:
            _f = dict(_f, p_model=_pmodel(_f))
        _ft = [dict(t, formation=dict(t["formation"], p_model=_pmodel(t.get("formation"))))
               if t.get("formation") else t for t in _ft]
        # 合成オッズは締切間近の更新（オッズが揃うタイミング）で入る。発売前は None
        if odds:
            _f = dict(_f, odds=synth_odds(_f, odds)) if _f else _f
            _ft = [dict(t, odds=synth_odds(t.get("formation"), odds)) for t in _ft]
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
            "formation": _f,
            # 「この展開でAが勝ったときの2着・3着」（ご要望の形）
            "conditional": conditional_orders(dist, lines, names),
            # ◎の置き場所ごとの買い目（◎頭/◎2着/◎3着/◎抜き）
            "form_types": _ft,
        })
    if not out:
        return None

    # ---- 展開をまたぐ統合買い目 ----
    # 分岐ごとに買うと点数が倍々になる（18点×2分岐=36点、重複を除いても30点前後）。
    # **混合分布に対して点数予算内でカバーを最大化**すれば、1つの買い目で
    # 全分岐をケアできる。どちらの展開に厚く張るかは P(展開) が自動で決める。
    # まず「分岐ごとに全部買った場合」＝長方形の和集合を出す。これが削減の基準。
    # 和集合は長方形2〜3枚ぶんなので、統合（長方形1枚）より表現力が高い。
    # 同じ点数なら和集合が有利になり得る一方、統合は1点あたりの効率が高い。
    uni: set = set()
    for b in out:
        f = b.get("formation") or {}
        uni |= {(a, b2, c) for a in f.get("first", []) for b2 in f.get("second", [])
                for c in f.get("third", []) if len({a, b2, c}) == 3}
    each = {"points": len(uni),
            "p_model": round(sum(mix.get(k, 0.0) for k in uni), 4)} if uni else None

    merged = build_plan(dists, mix, _pmodel, odds)

    return {
        "branches": out,
        # 1つの買い目で全分岐をケアする案（点数別）。each は分岐ごとに全部買った場合の参考値
        "merged": merged,
        "merged_each": each,
        "covered": round(sum(x["prob"] for x in out), 4),
        "hit_rate": 0.625,      # 主導権予測の out-of-sample 実測（記者先頭49.4%）
        "note": "主導権の確率は展開AI（男子62.5%的中）。分岐内の着順は実測の役割別シェアへ"
                "合わせた条件付き分布。買い目は展開の読み・紐選びの材料であって推奨ではない"
                "（期待値ゾーンは検証済みで存在しない: 全点均等買いROI 58.0%・上限75%）。",
    }

def conditional_orders(dist: dict[tuple, float], lines: list[list[int]],
                       names: dict[int, str] | None = None,
                       top_win: int = 3, top_n: int = 4) -> list[dict]:
    """「この展開で A が勝ったとき、2着・3着は誰か」を勝者ごとに返す。

    dist は展開を条件にした三連単分布。ここから 1着で条件付けるだけなので**追加のモデルは不要**。
      P(2着=x | 1着=a) / P(3着=y | 1着=a)  … いずれも dist の周辺化

    実測(25,235R)の構造が非対称なので、読み手に役割ラベルを添える:
      2着は勝者のライン内から **56.3%**（番手32.6 + 同ライン他23.7）
      3着は他ラインから **54%超**（他ライン番手26.2 + 他ライン先頭23.2 + 3番手4.8）
    → 2着はライン内で固め、3着は他ラインへ広げるのが構造的に正しい。
    """
    names = names or {}
    lo = {c: i for i, mem in enumerate(lines) for c in mem}

    def role(x: int, w: int) -> str:
        lw, lx = lo.get(w), lo.get(x)
        if lx is None or lw is None:
            return "単騎"
        if lx == lw:
            mem = lines[lw]
            i = mem.index(w)
            return "番手" if (i + 1 < len(mem) and mem[i + 1] == x) else "同ライン"
        mem = lines[lx]
        if len(mem) == 1:
            return "単騎"
        return "別線先頭" if mem.index(x) == 0 else (
            "別線番手" if mem.index(x) == 1 else "別線3番手")

    p1: dict[int, float] = {}
    for (a, _b, _c), p in dist.items():
        p1[a] = p1.get(a, 0.0) + p
    out = []
    for w, pw in sorted(p1.items(), key=lambda kv: -kv[1])[:top_win]:
        if pw <= 0:
            continue
        p2: dict[int, float] = {}
        p3: dict[int, float] = {}
        for (a, b, c), p in dist.items():
            if a != w:
                continue
            p2[b] = p2.get(b, 0.0) + p
            p3[c] = p3.get(c, 0.0) + p
        z = sum(p2.values()) or 1.0
        pack = lambda d: [{"car": x, "name": names.get(x), "role": role(x, w),
                           "prob": round(v / z, 4)}
                          for x, v in sorted(d.items(), key=lambda kv: -kv[1])[:top_n]]
        # ライン内/外の内訳。買い目の広げ方を決める材料
        inline2 = sum(v for x, v in p2.items() if lo.get(x) == lo.get(w)) / z
        inline3 = sum(v for x, v in p3.items() if lo.get(x) == lo.get(w)) / z
        out.append({
            "car": w, "name": names.get(w), "prob": round(pw, 4),
            "second": pack(p2), "third": pack(p3),
            "second_inline": round(inline2, 4),
            "third_inline": round(inline3, 4),
        })
    return out

# 買い目の型。◎を1着に固定する必要はない。実測では◎が1着を外す方が多い
# （男子の◎1着的中は43.4%＝**56.6%は◎が勝たない**）ので、◎を2着・3着に置く型と
# ◎を外す型を並べて出せるようにする。型ごとに「その形になる確率」も返す。
FORM_KINDS = ("◎頭", "◎2着", "◎3着", "◎抜き")


def _restrict(dist: dict[tuple, float], fav: int, kind: str) -> dict[tuple, float]:
    """型に合う組合せだけ残して正規化する（＝その型を条件にした分布）。"""
    if kind == "◎頭":
        sub = {k: v for k, v in dist.items() if k[0] == fav}
    elif kind == "◎2着":
        sub = {k: v for k, v in dist.items() if k[1] == fav}
    elif kind == "◎3着":
        sub = {k: v for k, v in dist.items() if k[2] == fav}
    else:                                    # ◎抜き＝◎が3着圏外
        sub = {k: v for k, v in dist.items() if fav not in k}
    z = sum(sub.values())
    return ({k: v / z for k, v in sub.items()}, z) if z > 0 else ({}, 0.0)


def formation_types(dist: dict[tuple, float], fav: int, budget: int = 18,
                    min_prob: float = 0.08) -> list[dict]:
    """◎の置き場所ごとに買い目を組む。確率 min_prob 未満の型は出さない。

    scenario_prob … その型になる確率（◎頭なら P(◎が1着)、◎抜きなら P(◎が3着圏外)）
    cover         … その型が起きたと仮定したときに買い目が当たる確率（条件付き）
    cover_abs     … 無条件で当たる確率 = scenario_prob × cover
    **いずれも回収率ではない。**
    """
    out = []
    for kind in FORM_KINDS:
        sub, z = _restrict(dist, fav, kind)
        if not sub or z < min_prob:
            continue
        f = _formation(sub, budget=budget)
        if not f:
            continue
        f = dict(f)
        # ◎の位置が固定される型は、その枠を◎1頭に固定して表記する
        if kind == "◎2着":
            f["second"] = [fav]
        elif kind == "◎3着":
            f["third"] = [fav]
        j = lambda xs: "".join(str(x) for x in sorted(xs))
        f["text"] = f"{j(f['first'])}-{j(f['second'])}-{j(f['third'])}"
        combos = {(a, b, c) for a in f["first"] for b in f["second"] for c in f["third"]
                  if len({a, b, c}) == 3}
        f["points"] = len(combos)
        f["cover"] = round(sum(sub.get(c, 0.0) for c in combos), 4)
        out.append({
            "kind": kind,
            "scenario_prob": round(z, 4),
            "cover": f["cover"],
            "cover_abs": round(z * f["cover"], 4),
            "formation": f,
        })
    return out

