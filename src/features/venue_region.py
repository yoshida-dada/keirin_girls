"""競輪場の所在地(都道府県)と競輪地区マップ。「地元開催」判定に使う。

用途:
    選手の登録府県(entries.prefecture) と 会場の所在府県 を突き合わせ、
    「地元開催（is_home_pref / is_home_district）」を発走前確定情報だけで判定する。

会場→所在県（2026-07-16 確認, venue_meta.py の会場名に対応）:
    各競輪場の所在都道府県は公知。京王閣(調布市)=東京, 弥彦=新潟 など。
県→地区は競輪の標準9地区運用を8グループに集約（本プロジェクトの層別粒度）:
    北日本 / 関東 / 南関東 / 中部 / 近畿 / 中国 / 四国 / 九州
    ※石川は中部、新潟・長野は関東、東京は南関東（競輪登録地の慣行に準拠）。

注意:
    prefecture/venue は発走前に確定する情報のみ（リーク無し）。results は使わない。
    entries.prefecture の '外国'(海外登録) は地区を持たないため常に非地元扱い。
"""
from __future__ import annotations

from src.features.venue_meta import venue_name

# venue_code(2桁) -> 所在都道府県
VENUE_PREF: dict[str, str] = {
    # --- 北日本 ---
    "11": "北海道",   # 函館
    "12": "青森",     # 青森
    "13": "福島",     # いわき平
    # --- 関東 ---
    "21": "新潟",     # 弥彦
    "22": "群馬",     # 前橋
    "23": "茨城",     # 取手
    "24": "栃木",     # 宇都宮
    "25": "埼玉",     # 大宮
    "26": "埼玉",     # 西武園
    # --- 南関東 ---
    "27": "東京",     # 京王閣（調布市）
    "28": "東京",     # 立川
    "31": "千葉",     # 松戸
    "34": "神奈川",   # 川崎
    "35": "神奈川",   # 平塚
    "36": "神奈川",   # 小田原
    "37": "静岡",     # 伊東温泉
    "38": "静岡",     # 静岡
    # --- 中部 ---
    "42": "愛知",     # 名古屋
    "43": "岐阜",     # 岐阜
    "44": "岐阜",     # 大垣
    "45": "愛知",     # 豊橋
    "46": "富山",     # 富山
    "47": "三重",     # 松阪
    "48": "三重",     # 四日市
    # --- 近畿 ---
    "51": "福井",     # 福井
    "53": "奈良",     # 奈良
    "54": "京都",     # 京都向日町
    "55": "和歌山",   # 和歌山
    "56": "大阪",     # 岸和田
    # --- 中国 ---
    "61": "岡山",     # 玉野
    "62": "広島",     # 広島
    "63": "山口",     # 防府
    # --- 四国 ---
    "71": "香川",     # 高松
    "73": "徳島",     # 小松島
    "74": "高知",     # 高知
    "75": "愛媛",     # 松山
    # --- 九州 ---
    "81": "福岡",     # 小倉
    "83": "福岡",     # 久留米
    "84": "佐賀",     # 武雄
    "85": "長崎",     # 佐世保
    "86": "大分",     # 別府
    "87": "熊本",     # 熊本
}

# 都道府県 -> 競輪地区（8グループ）
PREF_DISTRICT: dict[str, str] = {
    # 北日本
    "北海道": "北日本", "青森": "北日本", "岩手": "北日本", "宮城": "北日本",
    "秋田": "北日本", "山形": "北日本", "福島": "北日本",
    # 関東
    "茨城": "関東", "栃木": "関東", "群馬": "関東", "埼玉": "関東",
    "新潟": "関東", "長野": "関東",
    # 南関東
    "千葉": "南関東", "東京": "南関東", "神奈川": "南関東", "山梨": "南関東",
    "静岡": "南関東",
    # 中部
    "富山": "中部", "石川": "中部", "岐阜": "中部", "愛知": "中部", "三重": "中部",
    # 近畿
    "福井": "近畿", "滋賀": "近畿", "京都": "近畿", "大阪": "近畿",
    "兵庫": "近畿", "奈良": "近畿", "和歌山": "近畿",
    # 中国
    "鳥取": "中国", "島根": "中国", "岡山": "中国", "広島": "中国", "山口": "中国",
    # 四国
    "徳島": "四国", "香川": "四国", "愛媛": "四国", "高知": "四国",
    # 九州
    "福岡": "九州", "佐賀": "九州", "長崎": "九州", "熊本": "九州",
    "大分": "九州", "宮崎": "九州", "鹿児島": "九州", "沖縄": "九州",
}


def venue_pref(venue_code: str) -> str | None:
    """会場コードの所在都道府県。不明は None。"""
    return VENUE_PREF.get(venue_code)


def venue_district(venue_code: str) -> str | None:
    """会場コードの所在地区。不明は None。"""
    p = VENUE_PREF.get(venue_code)
    return PREF_DISTRICT.get(p) if p else None


def pref_district(pref: str | None) -> str | None:
    """選手の登録府県の地区。'外国'・不明は None。"""
    if not pref:
        return None
    return PREF_DISTRICT.get(pref.strip())


def is_home_pref(rider_pref: str | None, venue_code: str) -> bool:
    """選手の登録県 == 会場所在県 か。"""
    vp = VENUE_PREF.get(venue_code)
    return bool(rider_pref) and vp is not None and rider_pref.strip() == vp


def is_home_district(rider_pref: str | None, venue_code: str) -> bool:
    """選手の登録地区 == 会場所在地区 か。"""
    rd = pref_district(rider_pref)
    vd = venue_district(venue_code)
    return rd is not None and vd is not None and rd == vd


def describe_venue(venue_code: str) -> str:
    """'11 函館(北海道/北日本)' の形式で説明を返す（レポート用）。"""
    return f"{venue_code} {venue_name(venue_code)}({VENUE_PREF.get(venue_code)}/{venue_district(venue_code)})"
