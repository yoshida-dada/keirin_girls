"""ライン特徴（男子競輪）。並び予想のライン構造をモデル入力列にする。

ガールズにはライン概念が無いため、これは男子専用の特徴群。実測（25,134R・直近1年）で
確認した構造を素直に列にする（詳細は men_keirin_plan.md 4.6節）:

  ライン強度 × ライン内位置（7車・1着率/3着内率）
    1位ライン  先頭 33.2%/66%  番手 22.4%/65%  3番手 3.7%/44%  単騎 18.6%/47%
    3位ライン  先頭 11.3%/33%  番手  8.0%/30%  3番手 1.8%/21%  単騎  4.8%/20%
  - 1着は「強いラインの先頭」に集中（ランダム14.3%の2.3倍）
  - 3着内では最強ラインの番手(65%)が先頭(66%)とほぼ同格＝紐として同価値
  - 3番手はどのラインでも1着が無い(1.8〜3.7%)
  - ライン長は1人あたりで見ると3名以上で有利、2名はランダム未満(0.87倍)

**級班との交互作用が要る。** 番手の価値が級班で反転するため:
    番手÷先頭の1着率比   S1/S2 1.09（番手の方が強い） → A3 0.47（先頭の半分以下）
  原因は「番手が千切れる」ではなく「先頭が逃げ切る」（逃げ決着 A3 35.6% vs S1/S2 15.5%）。
  級班はレース定数で順位モデル（シフト不変）にそのまま入れても効かないので、
  レース内で変動する「ライン内位置」に掛けた交互作用として注入する。
"""
from __future__ import annotations

# モデル入力の列名（この順で連結する）
LINE_KEYS = [
    "ln_head",          # 2名以上のラインの先頭なら1
    "ln_mate",          # 番手なら1
    "ln_third",         # 3番手以降なら1
    "ln_solo",          # 単騎なら1
    "ln_len_rel",       # 所属ライン長 − レース内の平均ライン長
    "ln_str_rel",       # 所属ラインの平均競走得点 − レース内の全ライン平均
    "x_mate_class",     # ln_mate × 級班水準（番手の価値の反転を表す）
    "x_head_class",     # ln_head × 級班水準（A3ほど先頭が強いことを表す）
]
# 並び予想の脚質 → 前がかり度。**実測のB(主導権)取得率で序列を決めた**（推測しない）:
#   男子   先行55.1% 押え先34.4% カマシ31.0% / 自在11.5% まくり7.6% イン待6.3% /
#          追上1.2% 競り0.3% 追込0.3%
#   ガールズ カマシ43.4% 先行42.6% 押え先39.2% / まくり21.7% 自在12.3% / 追上1.6% 追込0.7%
# 「まくり」は直感では自力型だが実測では低い（後方から捲る宣言＝バックは取らない）ので1にした。
# **rider_narabi.LEG_AGGR とは別物**。あちらはガールズ本番モデルが学習時に使った値で、
# 書き換えると学習と推論で値が食い違う（追上33.9%が未定義のまま1.0扱い＝既知の不具合だが、
# 直すには再学習が必要）。男子はここで正しい値を使う。
LEG_AGGR_MEN = {
    "先行": 2.0, "押え先": 2.0, "カマシ": 2.0,
    "自在": 1.0, "まくり": 1.0, "捲り": 1.0, "イン待": 0.5, "イン切": 0.5,
    "追上": 0.0, "追込": 0.0, "競り": 0.0, "差し": 0.0, "マーク": 0.0, "追": 0.0,
}
# 脚質の前がかり度（ライン内位置では捉えられない「その選手の意図」）
LEG_KEYS = ["ln_leg"]


def leg_aggr(leg) -> float:
    """脚質 → 前がかり度。未知語は 1.0（自在相当）に寄せる。"""
    return LEG_AGGR_MEN.get(leg, 1.0)

# 級班 → 水準。S級=2 / A1・A2=1 / A3=0 とし、レース平均から CLASS_CENTER を引いて中心化する。
# 中心化後は S級レース≒+1 / A1A2≒0 / A3≒-1 になり、番手の価値が上がる向きと符号が揃う。
CLASS_LEVEL = {"SS": 2.0, "S1": 2.0, "S2": 2.0, "A1": 1.0, "A2": 1.0, "A3": 0.0, "L1": 1.0}
CLASS_CENTER = 1.0


def class_level(class_ranks) -> float:
    """出走選手の級班からレースの級班水準（中心化済み）を返す。不明は0（＝交互作用なし）。"""
    vals = [CLASS_LEVEL[c] for c in (class_ranks or []) if c in CLASS_LEVEL]
    if not vals:
        return 0.0
    return sum(vals) / len(vals) - CLASS_CENTER


def line_columns(cars: list[int], line_of: dict[int, tuple[int, int]],
                 scores: dict[int, float], cls_level: float) -> dict[int, list]:
    """出走車 → ライン特徴列（LINE_KEYS 順）。

    line_of : {車番: (line_id, pos_in_line)}。並び予想が無い車は欠落してよい
    scores  : {車番: 競走得点}
    cls_level: class_level() の戻り値（レース定数）

    並び予想が全く無いレースでは全列0（＝特徴なし）になり、学習・推論とも落ちない。
    """
    members: dict[int, list[int]] = {}
    for car, (li, _pi) in line_of.items():
        members.setdefault(li, []).append(car)

    lens = [len(v) for v in members.values()]
    mean_len = sum(lens) / len(lens) if lens else 0.0
    # ライン平均得点（そのラインの構成員の得点平均）。レース内の全ライン平均を引いて相対化する
    lstr = {li: (sum(scores.get(c, 0.0) for c in mem) / len(mem)) for li, mem in members.items()}
    mean_str = sum(lstr.values()) / len(lstr) if lstr else 0.0

    out: dict[int, list] = {}
    for car in cars:
        li_pi = line_of.get(car)
        if li_pi is None:
            out[car] = [0.0] * len(LINE_KEYS)
            continue
        li, pi = li_pi
        n = len(members.get(li, []))
        head = 1.0 if (n >= 2 and pi == 0) else 0.0
        mate = 1.0 if (n >= 2 and pi == 1) else 0.0
        third = 1.0 if (n >= 2 and pi >= 2) else 0.0
        solo = 1.0 if n == 1 else 0.0
        out[car] = [
            head, mate, third, solo,
            n - mean_len,
            lstr.get(li, 0.0) - mean_str,
            mate * cls_level,
            head * cls_level,
        ]
    return out
