"""バンクの諸元と統計的な特徴を、表示用に1レース分まとめる。

構成:
  - 静的諸元（周長・みなし直線・カント・幅員）… BANK_SPEC。出典は keirin-brother.com
    /race-track/（2026-07-29取得）。みなし直線は Wikipedia「競輪場」と全場一致を確認。
    広島(57.9m)はWikipedia未記載で本ソースのみ。千葉(PIST6)はガールズ対象外。
  - 相対的な位置づけ … 同一周長の会場内で直線・カントが長い/短い、急/緩を順位から判定。
  - 統計的な有利脚質 … kimarite_stats.json のバンク別実測（各n千単位で安定）を根拠にする。
    会場別の実測も併記するが**参考値**（n=40〜292と薄い。会場別重みは予測用途では
    検証の結果不採用＝scripts/validate_venue_interaction.py）。

**重要**: カント・みなし直線は決まり手の予測には使えないことが検証済み
（同一周長内で |r|<0.2、scripts/analyze_bank_specs.py）。ここでの用途は
「このバンクはどういう形か」を読み手に伝える説明であって、予測根拠ではない。

DBに触らないので refresh_predictions.py（DB非依存）からも安全に呼べる。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.features import venue_meta as vm

STATS_PATH = Path(__file__).resolve().parent.parent / "model" / "kimarite_stats.json"


def _dms(s: str) -> float:
    m = re.match(r"(\d+)°(\d+)['′´](\d+)", s)
    d, mi, se = (int(x) for x in m.groups())
    return round(d + mi / 60 + se / 3600, 2)


# 会場名 -> (みなし直線m, センター部カント, 直線部傾斜, ホーム幅, バック幅, センター幅)
_RAW: dict[str, tuple] = {
    "函館": (51.3, "30°36'51\"", "3°26'1\"", 10.8, 9.8, 7.8),
    "青森": (58.9, "32°15'07\"", "2°51'45\"", 10.8, 9.8, 7.8),
    "いわき平": (62.7, "32°54'45\"", "3°26'1\"", 10.0, 10.0, 7.3),
    "弥彦": (63.1, "32°24'17\"", "2°51'45\"", 10.1, 9.0, 7.3),
    "前橋": (46.7, "36°0'0\"", "4°0'0\"", 9.9, 9.9, 9.9),
    "取手": (54.8, "31°30'25\"", "2°51'44\"", 10.0, 10.0, 7.5),
    "宇都宮": (63.3, "25°47'44\"", "2°51'44\"", 10.3, 11.3, 8.3),
    "大宮": (66.7, "26°16'40\"", "3°26'1\"", 10.3, 9.3, 7.5),
    "西武園": (47.6, "29°26'54\"", "2°51'45\"", 11.0, 10.0, 7.5),
    "京王閣": (51.5, "32°10'34\"", "2°51'44\"", 10.3, 9.0, 7.5),
    "立川": (58.0, "31°13'6\"", "2°17'27\"", 9.7, 8.7, 7.7),
    "松戸": (38.2, "29°44'42\"", "3°1'2\"", 11.1, 9.6, 8.1),
    "川崎": (58.0, "32°10'14\"", "3°26'1\"", 10.3, 9.3, 8.3),
    "平塚": (54.2, "31°28'37\"", "3°26'1\"", 11.0, 9.3, 7.5),
    "小田原": (36.1, "35°34'12\"", "3°26'1\"", 11.3, 9.0, 7.5),
    "伊東温泉": (46.6, "34°41'9\"", "3°26'1\"", 11.0, 9.3, 7.8),
    "静岡": (56.4, "30°43'22\"", "2°51'45\"", 10.3, 9.3, 7.5),
    "名古屋": (58.8, "34°1'47\"", "2°51'45\"", 10.3, 9.3, 7.3),
    "岐阜": (59.3, "32°15'7\"", "2°51'45\"", 10.2, 9.0, 7.4),
    "大垣": (56.0, "30°37'8\"", "2°51'45\"", 10.2, 9.0, 7.4),
    "豊橋": (60.3, "33°50'22\"", "2°17'26\"", 10.3, 9.3, 7.8),
    "富山": (43.0, "33°41'24\"", "3°26'1\"", 10.2, 9.2, 6.4),
    "松阪": (61.5, "34°25'29\"", "2°51'45\"", 10.9, 9.0, 7.7),
    "四日市": (62.4, "32°15'7\"", "2°51'45\"", 13.3, 11.5, 8.5),
    "福井": (52.8, "31°28'37\"", "2°51'45\"", 10.5, 9.0, 7.5),
    "奈良": (38.0, "33°25'47\"", "4°51'48\"", 10.8, 7.8, 7.8),
    "京都向日町": (47.3, "30°29'7\"", "3°26'1\"", 10.3, 9.3, 7.6),
    "和歌山": (59.9, "32°15'7\"", "2°51'45\"", 11.4, 9.3, 7.7),
    "岸和田": (56.7, "30°56'0\"", "2°51'45\"", 10.2, 10.1, 7.3),
    "玉野": (47.9, "30°37'33\"", "3°26'1\"", 10.3, 9.3, 7.5),
    "広島": (57.9, "32°31'40\"", "3°26'1\"", 10.5, 8.5, 7.3),
    "防府": (42.5, "34°41'9\"", "4°34'26\"", 10.2, 9.1, 7.4),
    "高松": (54.8, "33°15'50\"", "2°51'45\"", 11.0, 9.0, 8.0),
    "小松島": (55.5, "29°46'27\"", "2°51'45\"", 10.3, 9.3, 8.3),
    "高知": (52.0, "24°29'51\"", "3°26'1\"", 11.3, 10.8, 7.8),
    "松山": (58.6, "34°1'48\"", "2°51'45\"", 10.3, 9.3, 7.3),
    "小倉": (56.9, "34°1'48\"", "3°26'1\"", 11.0, 10.0, 8.0),
    "久留米": (50.7, "31°28'37\"", "3°26'1\"", 11.0, 10.0, 9.0),
    "武雄": (64.4, "32°0'19\"", "2°17'26\"", 9.7, 8.7, 7.4),
    "佐世保": (40.2, "31°28'37\"", "3°26'1\"", 10.0, 9.0, 7.5),
    "別府": (59.9, "33°41'24\"", "2°51'45\"", 10.0, 9.0, 8.0),
    "熊本": (60.3, "34°15'29\"", "2°51'45\"", 10.0, 9.0, 8.0),
}
BANK_SPEC = {k: {"straight": v[0], "cant": _dms(v[1]), "cant_straight": _dms(v[2]),
                 "w_home": v[3], "w_back": v[4], "w_center": v[5]} for k, v in _RAW.items()}

BANK_LABEL = {333: "短走路", 400: "標準走路", 500: "長走路"}
# 「有利脚質」と書く最小の差（pt）。これ未満は全体平均並みとして脚質を名指ししない。
MIN_ADV_DIFF = 3.0


@lru_cache(maxsize=1)
def _stats() -> dict:
    try:
        return json.loads(STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _by_bank() -> dict[int, list[tuple[str, dict]]]:
    """周長ごとの会場リスト（同一周長内での相対位置を出すため）。"""
    out: dict[int, list] = {}
    for code, meta in vm.VENUE.items():
        nm, bank = meta.get("name"), meta.get("bank")
        sp = BANK_SPEC.get(nm or "")
        if sp and bank:
            out.setdefault(bank, []).append((nm, sp))
    return out


def _rel_pos(bank: int, name: str, key: str) -> tuple[int, int]:
    """同一周長内での順位（1=最大）と母数。"""
    peers = _by_bank().get(bank) or []
    ranked = sorted(peers, key=lambda kv: -kv[1][key])
    for i, (nm, _) in enumerate(ranked, 1):
        if nm == name:
            return i, len(ranked)
    return 0, len(ranked)


def _band(rank: int, total: int, hi: str, lo: str, mid: str = "標準的") -> str:
    if not rank or total < 4:
        return mid
    if rank == 1:
        return f"同周長で最も{hi}"
    if rank <= max(2, total // 4):
        return hi
    if rank > total - max(2, total // 4):
        return lo if rank < total else f"同周長で最も{lo}"
    return mid


def profile(venue_code: str | None) -> dict | None:
    """1会場ぶんのバンク特性を返す。諸元が無ければ None。"""
    if not venue_code:
        return None
    name, bank = vm.venue_name(venue_code), vm.bank_length(venue_code)
    sp = BANK_SPEC.get(name or "")
    if not sp or not bank:
        return None

    r_st, n_st = _rel_pos(bank, name, "straight")
    r_ca, n_ca = _rel_pos(bank, name, "cant")
    traits = [BANK_LABEL.get(bank, f"{bank}m")]
    traits.append(f"直線 {_band(r_st, n_st, '長い', '短い')}（同周長{n_st}場中{r_st}位）")
    traits.append(f"カント {_band(r_ca, n_ca, '急', '緩い')}（同{n_ca}場中{r_ca}位）")

    st = _stats()
    g = ((st.get("global") or {}).get("kim") or {})
    bank_kim = ((st.get("bank") or {}).get(str(bank)) or {})
    ven = ((st.get("venue") or {}).get(venue_code) or {})

    adv = None
    bk = bank_kim.get("kim") or {}
    if bk and g:
        # 有利脚質はバンク別実測（n千単位で安定）を根拠にする。会場別は薄いので使わない。
        diffs = {k: bk.get(k, 0) - g.get(k, 0) for k in ("逃", "捲", "差")}
        top = max(diffs, key=diffs.get)
        jp = {"逃": "先行（逃げ）", "捲": "捲り", "差": "差し"}[top]
        ratio = (bk.get(top, 0) / g[top]) if g.get(top) else 1.0
        if diffs[top] < MIN_ADV_DIFF:
            # 400mは全レースの約8割を占めるため全体平均とほぼ一致する。1pt程度の差を
            # 「有利」と書くと実態以上に読まれるので、際立たない場合はそう明示する。
            adv = {"leg": None, "diff": round(diffs[top], 1),
                   "text": "際立った有利脚質はない（全体平均並みの決まり手構成）"}
        else:
            adv = {"leg": jp, "rate": bk.get(top), "global": g.get(top),
                   "diff": round(diffs[top], 1), "ratio": round(ratio, 2),
                   "text": f"{jp}が有利（{bk.get(top)}% / 全体{g.get(top)}% ＝{ratio:.2f}倍）"}

    return {
        "venue": name, "bank": bank, "bank_label": BANK_LABEL.get(bank),
        "straight": sp["straight"], "cant": sp["cant"],
        "w_home": sp["w_home"],
        "traits": traits,
        "bank_kimarite": bk or None,
        "bank_n": bank_kim.get("n"),
        "venue_kimarite": (ven.get("kim") or None),
        "venue_n": ven.get("n"),
        "b_reliability": ({"rentai": bank_kim["b_rentai"], "gaiji": bank_kim["b_gaiji"]}
                          if bank_kim.get("b_rentai") is not None else None),
        "advantage": adv,
        "note": "諸元は競輪場データ（みなし直線はWikipediaと照合）。有利脚質はバンク別の実測"
                "決まり手（全期間）に基づく。会場別の数値は標本が薄く参考値。"
                "カント・直線は決まり手の予測には使えないことが検証済み（同一周長内で無相関）。",
    }
