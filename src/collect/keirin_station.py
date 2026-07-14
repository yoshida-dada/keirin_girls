"""競輪ステーション 選手詳細パーサ（S1・選手成績詳細。長期＋直近4ヶ月）。

URL: https://keirin-station.com/keirindb/player/detail/{登録番号}/
robots.txt は一般UAに /keirindb/ を許可（禁止は /system/ 等のみ）。

このページから抽出（まずは直近4ヶ月集計＋年度別通算 = ユーザー方針）:
  プロフィール : 登録番号 / 名前 / 府県 / 期別 / 級班
  直近4ヶ月    : 出走回数・B・H・S / 勝率・2連対率・3連対率 / 着順(1/2/3/着外/棄権/失格)
                 決まり手(逃げ/捲り/差し/マーク %) / 競走得点(平均/最高/最低)
  通算成績     : 〜1昨年 / 昨年 / 本年 / 通算 × 出走数・優勝・1〜3着

★タイムは「競輪学校時代の記録」のみで直近レースのタイムではない（本パーサでは扱わない）。
★バンク別勝率テーブルは無い（得意バンクは定性表現）。バンク別はオッズパーク/最近の成績で補完。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from src.collect.base import fetch

HOST = "https://keirin-station.com"


@dataclass
class PlayerStats:
    rider_id: str                  # 登録番号
    name: str = ""
    prefecture: str = ""
    term: int | None = None
    class_rank: str = ""
    # 直近4ヶ月
    starts: int | None = None
    back_count: int | None = None      # B
    home_count: int | None = None      # H
    start_count: int | None = None     # S
    win_rate: float | None = None      # 勝率(0-1)
    top2_rate: float | None = None     # 2連対率
    top3_rate: float | None = None     # 3連対率
    score_avg: float | None = None     # 競走得点 平均
    score_max: float | None = None
    score_min: float | None = None
    escape_rate: float | None = None   # 逃げ(0-1)
    dash_rate: float | None = None     # 捲り
    closing_rate: float | None = None  # 差し
    mark_rate: float | None = None     # マーク
    first: int | None = None
    second: int | None = None
    third: int | None = None
    out: int | None = None             # 着外
    dnf: int | None = None             # 棄権
    dsq: int | None = None             # 失格
    # 通算（回数）
    career: dict = field(default_factory=dict)  # {'total_starts','total_wins','year_starts',...}


def build_player_url(rider_id: str) -> str:
    return f"{HOST}/keirindb/player/detail/{rider_id}/"


def _num(s: str) -> float | None:
    """"2,452 回" -> 2452 / "13.30%" -> 0.133 / "110.57" -> 110.57。取れなければNone。"""
    if not s:
        return None
    s = s.replace(",", "").strip()
    pct = s.endswith("%")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m.group(0))
    return v / 100.0 if pct else v


def _int(s: str) -> int | None:
    v = _num(s)
    return int(v) if v is not None else None


def _find_label(soup: BeautifulSoup, label: str) -> str:
    """全テーブル横断で th==label の直後td текストを返す（プロフィールが複数表に分かれるため）。"""
    for th in soup.find_all("th"):
        if th.get_text(strip=True) == label:
            td = th.find_next_sibling("td")
            if td is not None:
                return td.get_text(" ", strip=True)
    return ""


def normalize_prefecture(pref: str) -> str:
    """"徳島県"/"東京都"/"大阪府"/"北海道" → "徳島"/"東京"/"大阪"/"北海道"。突合用の正規化。"""
    pref = pref.strip().replace("　", "")
    if pref == "北海道":
        return pref
    return re.sub(r"[都道府県]$", "", pref)


def _section_table(soup: BeautifulSoup, keyword: str):
    """見出し(h2/h3)にkeywordを含む直後のtableを返す。"""
    for h in soup.find_all(["h2", "h3"]):
        if keyword in h.get_text():
            return h.find_next("table")
    return None


def _kv(table) -> dict[str, str]:
    """label(th) -> value(直後のtd) の辞書。1行に複数ペアが並ぶ表に対応。"""
    out: dict[str, str] = {}
    if table is None:
        return out
    for th in table.find_all("th"):
        td = th.find_next_sibling("td")
        if td is not None:
            out[th.get_text(" ", strip=True)] = td.get_text(" ", strip=True)
    return out


def parse_player_detail(html: str, rider_id: str = "") -> PlayerStats:
    soup = BeautifulSoup(html, "html.parser")
    ps = PlayerStats(rider_id=rider_id)

    # プロフィール（登録番号を含む先頭付近の表）
    prof = _kv(soup.find("table"))
    for tb in soup.find_all("table"):
        kv = _kv(tb)
        if "登録番号" in kv:
            prof = kv
            break
    if not ps.rider_id:
        ps.rider_id = prof.get("登録番号", "")
    ps.prefecture = prof.get("府県", "")            # 生値（例 "徳島県"）。突合時に normalize_prefecture
    ps.class_rank = _find_label(soup, "級班")       # 期別/級班は別表にあるため横断取得
    ps.term = _int(_find_label(soup, "期別"))
    h1 = soup.find("h1")
    if h1:
        ps.name = h1.get_text(strip=True).replace("選手", "").strip()

    # 直近4ヶ月: 出走成績
    r = _kv(_section_table(soup, "出走成績"))
    ps.starts = _int(r.get("出走回数", ""))
    ps.back_count = _int(r.get("バック回数", ""))
    ps.home_count = _int(r.get("ホーム回数", ""))
    ps.start_count = _int(r.get("スタート回数", ""))
    ps.win_rate = _num(r.get("勝率", ""))
    ps.top2_rate = _num(r.get("2連対率", ""))
    ps.top3_rate = _num(r.get("3連対率", ""))

    # 競走得点
    sc = _kv(_section_table(soup, "競走得点の推移"))
    ps.score_avg = _num(sc.get("平均", ""))
    ps.score_max = _num(sc.get("最高", ""))
    ps.score_min = _num(sc.get("最低", ""))

    # 着順の推移
    o = _kv(_section_table(soup, "着順の推移"))
    ps.first = _int(o.get("1着", ""))
    ps.second = _int(o.get("2着", ""))
    ps.third = _int(o.get("3着", ""))
    ps.out = _int(o.get("着外", ""))
    ps.dnf = _int(o.get("棄権", ""))
    ps.dsq = _int(o.get("失格", ""))

    # 決まり手（率）
    k = _kv(_section_table(soup, "決まり手"))
    ps.escape_rate = _num(k.get("逃げ", ""))
    ps.dash_rate = _num(k.get("捲り", ""))
    ps.closing_rate = _num(k.get("差し", ""))
    ps.mark_rate = _num(k.get("マーク", ""))

    # 通算成績（マトリクス: 行ラベル × 〜1昨年/昨年/本年/通算）
    ps.career = _parse_career(_section_table(soup, "通算成績"))
    return ps


def _parse_career(table) -> dict:
    """通算成績テーブルを {行ラベル: {列: 値}} で返す。列は 〜1昨年/昨年/本年/通算。"""
    if table is None:
        return {}
    rows = table.find_all("tr")
    if not rows:
        return {}
    cols = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"]) if c.get_text(strip=True)]
    out: dict = {}
    for tr in rows[1:]:
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        label = cells[0].get_text(strip=True)
        vals = [_int(c.get_text(strip=True)) for c in cells[1:]]
        out[label] = dict(zip(cols, vals))
    return out


def fetch_player_detail(rider_id: str) -> PlayerStats:
    """選手詳細を取得しパースする（ネットワークアクセスあり）。"""
    res = fetch(build_player_url(rider_id))
    return parse_player_detail(res.text, rider_id=rider_id)


# ============================================================
# 選手検索（GambooBET出走表 → 登録番号の名前突合）
# ============================================================

@dataclass
class RiderCandidate:
    rider_id: str
    name: str            # 漢字氏名（空白除去済み 例 "青木美保"）
    prefecture: str      # 府県（サフィックス無し 例 "埼玉"）
    term: int | None     # 卒業期＝期別
    class_rank: str      # 例 "Ｌ級１班"


def _strip_spaces(s: str) -> str:
    return re.sub(r"\s|　", "", s or "")


def build_search_url(name_1: str = "", name_2: str = "", *, girls: bool = True,
                     active: bool = True) -> str:
    """選手検索URL。name_1=姓, name_2=名。girls=Trueでガールズに絞る。"""
    from urllib.parse import urlencode
    params = {
        "player_profile[name_1]": name_1,
        "player_profile[name_2]": name_2,
        "submit[btn][player_profile][get]": "この条件で検索する",
    }
    if girls:
        params["player_profile[girls_flag]"] = "1"
    if active:
        params["player_profile[active_flag]"] = "1"
    return f"{HOST}/keirindb/search/player/?" + urlencode(params)


def parse_search_results(html: str) -> list[RiderCandidate]:
    """検索結果HTMLから候補一覧を返す。1選手=名前行(anchor)＋直後の登録番号行の2行構成。"""
    soup = BeautifulSoup(html, "html.parser")
    out: list[RiderCandidate] = []
    for a in soup.find_all("a", href=re.compile(r"/keirindb/player/detail/(\d+)/")):
        rid = re.search(r"/detail/(\d+)/", a["href"]).group(1)
        name_row = a.find_parent("tr")
        if name_row is None:
            continue
        name_cells = [c.get_text(" ", strip=True) for c in name_row.find_all(["th", "td"])]
        class_rank = name_cells[2] if len(name_cells) > 2 else ""
        term = None
        for c in name_cells:
            m = re.match(r"(\d+)期", c)
            if m:
                term = int(m.group(1))
                break
        id_row = name_row.find_next_sibling("tr")
        prefecture = ""
        if id_row:
            id_cells = [c.get_text(" ", strip=True) for c in id_row.find_all(["th", "td"])]
            if len(id_cells) > 1:
                prefecture = id_cells[1]
        out.append(RiderCandidate(rider_id=rid, name=_strip_spaces(a.get_text(" ", strip=True)),
                                  prefecture=normalize_prefecture(prefecture), term=term,
                                  class_rank=class_rank))
    return out


def match_rider(name: str, prefecture: str, term: int | None,
                candidates: list[RiderCandidate]) -> RiderCandidate | None:
    """出走表の (名前, 府県, 期別) を候補に突合し登録番号を確定する。

    名前(空白除去)一致を必須とし、府県・期別で絞る。1件に絞れなければ None（要人手確認）。
    """
    key = _strip_spaces(name)
    pref = normalize_prefecture(prefecture)
    named = [c for c in candidates if c.name == key]
    if len(named) == 1:
        return named[0]
    narrowed = [c for c in named if c.prefecture == pref and (term is None or c.term == term)]
    if len(narrowed) == 1:
        return narrowed[0]
    return None


def resolve_rider_id(name: str, prefecture: str, term: int | None, *,
                     girls: bool = True) -> str | None:
    """名前で検索し登録番号を返す（ネットワークアクセスあり）。姓名は空白で分割。"""
    parts = name.split()
    name_1 = parts[0] if parts else name
    name_2 = parts[1] if len(parts) > 1 else ""
    res = fetch(build_search_url(name_1, name_2, girls=girls))
    cand = parse_search_results(res.text)
    m = match_rider(name, prefecture, term, cand)
    return m.rider_id if m else None
