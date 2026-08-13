"""男子のライン別の強さを数値化する（表示用）。

**数値はモデルの三連単確率から直接積み上げる。** 別の物差し（得点合計など）を持ち込むと
表示のライン評価と実際の買い目確率が食い違い、読み手が混乱するため。
1着確率はΣ=1、ライン決着確率も同じ分布から出すので内部矛盾しない。

出す指標（すべてそのレースのモデル分布から算出）:
  p_win      : このラインから1着が出る確率（メンバーの1着確率の和）
  p_12       : 1着・2着をこのラインで独占する確率（＝ライン決着）
  p_top3_any : 3着以内に1名以上入る確率
  score_avg  : 競走得点の平均（並びの見当をつける補助。予測には使っていない）

実測の裏づけ（men_keirin_plan.md 4.6, 男子25,134R）:
  1着は「強いラインの先頭」に集中する（強度1位の先頭 33.2%＝ランダム14.3%の2.3倍）。
  3着内では最強ラインの番手(65%)が先頭(66%)とほぼ同格＝**紐として同価値**。
  3番手はどのラインでも1着がほぼ無い(1.8〜3.7%)が、最強ラインなら3着内44%で紐にはなる。
"""
from __future__ import annotations

POS_LABEL = {0: "先頭", 1: "番手", 2: "3番手", 3: "4番手"}


def _pos_label(pos: int | None, size: int) -> str:
    if size == 1:
        return "単騎"
    if pos is None:
        return "—"
    return POS_LABEL.get(pos, f"{pos + 1}番手")


def build_lines(lines: list[list[int]], strengths: dict[int, float],
                probs: dict[tuple, float] | None,
                names: dict[int, str] | None = None,
                scores: dict[int, float] | None = None,
                legs: dict[int, str] | None = None,
                classes: dict[int, str] | None = None) -> dict | None:
    """並び予想のライン構成 → ライン別の強さ。lines が空なら None（ガールズ）。

    probs は三連単の全通り確率 {(1着,2着,3着): p}。無ければ p_12 / p_top3_any は None。
    """
    if not lines:
        return None
    names = names or {}
    scores = scores or {}
    legs = legs or {}
    classes = classes or {}
    line_of = {car: li for li, mem in enumerate(lines) for car in mem}

    p12 = [0.0] * len(lines)          # 1-2着を同一ラインで独占
    top3_any = [0.0] * len(lines)     # 3着以内に1名以上
    if probs:
        for (a, b, c), p in probs.items():
            la, lb = line_of.get(a), line_of.get(b)
            if la is not None and la == lb:
                p12[la] += p
            hit = set()
            for car in (a, b, c):
                li = line_of.get(car)
                if li is not None:
                    hit.add(li)
            for li in hit:
                top3_any[li] += p

    out = []
    for li, mem in enumerate(lines):
        sc = [scores[c] for c in mem if scores.get(c) is not None]
        out.append({
            "line_id": li,
            "size": len(mem),
            "cars": [{"car": c, "name": names.get(c), "pos": i,
                      "pos_label": _pos_label(i, len(mem)),
                      "leg": legs.get(c), "class_rank": classes.get(c),
                      "win_prob": round(strengths.get(c, 0.0), 4)} for i, c in enumerate(mem)],
            "p_win": round(sum(strengths.get(c, 0.0) for c in mem), 4),
            "p_12": round(p12[li], 4) if probs else None,
            "p_top3_any": round(min(top3_any[li], 1.0), 4) if probs else None,
            "score_avg": round(sum(sc) / len(sc), 1) if sc else None,
        })
    # 1着確率の降順＝そのレースでの強さ順。rank は表示の「①②③」に使う
    order = sorted(range(len(out)), key=lambda i: -out[i]["p_win"])
    for rank, i in enumerate(order, 1):
        out[i]["rank"] = rank
    settle = round(sum(x["p_12"] for x in out), 4) if probs else None
    return {
        "lines": out,
        # ライン決着確率＝1・2着が同一ラインで決まる確率。紐をライン内で固めるか
        # 別ラインへ散らすかの判断材料になる（実測の全体平均は 1-2着同一ライン 約55〜60%）
        "settle_prob": settle,
        "note": "数値はこのレースのモデル分布から算出（1着確率の和・ライン決着確率）。"
                "実測では1着は強いラインの先頭に集中し、3着内では最強ラインの番手が"
                "先頭とほぼ同格＝紐として同価値。3番手は1着がほぼ無い。",
    }
