"""選手ごとの「ライン内位置別成績」と戦法系の実績（表示用）。

出走表に出す**説明用の統計**であってモデル入力ではない。`as_of` より前のレースだけを
集計する（CLAUDE.md 5「選手成績は as_of で結合」）。表示専用でも、過去レースの画面に
そのレース以降の成績が混じると実績表示として誤りになるため。

**何が出せて何が出せないか**（2026-08-16 に男子25,392Rで実測）:
  出せる（選手あたりn 中央値）
    位置別 先頭29 / 番手32 / 3番手以降19 / 単騎7   … 十分
    ライン分断 先頭視点28 / 番手視点31             … nは十分だが定義は近似（下記）
  薄い（既定では n>=5 でだけ表示する。MIN_N）
    かまし成功 中央値4（n>=10 は824人中64人）
    つっぱり成功 中央値3（n>=10 は760人中16人）
    競り勝ち 中央値1（n>=10 は2,258人中1人。競りは全体の1.5%しかない）
  出せない
    **飛びつき成功率**。結果に保存しているのは着順・決まり手・S/B印だけで、
    レース中の位置取り推移が無い。代替指標も作れないので項目自体を置かない。

**ライン分断率は「ちぎり率」ではない**。先頭が3着内かつ番手が着外だったレースの率で、
意図的に切ったのか、番手が力尽きたのか、他ラインに阻まれたのかは区別できない。
先頭視点と番手視点は同じ事象を数えているので全体率は一致する（実測とも18.8%で一致）。
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from functools import lru_cache

MIN_N = 5                     # これ未満は率を出さない（分母だけ返す）
POS_KEYS = ["先頭", "番手", "3番手以降", "単騎"]


def _pos_label(pos_in_line: int, size: int) -> str:
    if size <= 1:
        return "単騎"
    return "先頭" if pos_in_line == 0 else ("番手" if pos_in_line == 1 else "3番手以降")


def _rate(k: int, n: int) -> float | None:
    """n が MIN_N 未満なら率を出さない。分母は別に返すので「—(0/3)」と書ける。"""
    return round(k / n, 4) if n >= MIN_N else None


@lru_cache(maxsize=8)
def compute_line_stats(db_path: str, as_of: str | None = None) -> dict:
    """選手名 → 位置別成績と戦法系の実績。as_of（YYYY-MM-DD）より前のレースだけ集計。

    返り値: {選手名: {"pos": {位置: {...}}, "kamashi": {...}, ...}}
    """
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    where = " WHERE r.race_date < ?" if as_of else ""
    args = (as_of,) if as_of else ()

    nar: dict = defaultdict(dict)
    for rid, car, li, pi, leg, sg in c.execute(
            "SELECT n.race_id,n.car_number,n.line_id,n.pos_in_line,n.leg,n.seri_group"
            " FROM narabi n JOIN races r ON r.race_id=n.race_id" + where, args):
        nar[rid][car] = (li, pi, leg, sg)
    res: dict = defaultdict(dict)
    for rid, pos, car, sb in c.execute(
            "SELECT s.race_id,s.position,s.car_number,s.sb"
            " FROM results s JOIN races r ON r.race_id=s.race_id" + where, args):
        res[rid][car] = (pos, sb)
    nm: dict = {}
    for rid, car, n2 in c.execute(
            "SELECT e.race_id,e.car_number,e.rider_name"
            " FROM entries e JOIN races r ON r.race_id=e.race_id" + where, args):
        nm[(rid, car)] = n2
    c.close()

    acc: dict = defaultdict(lambda: {
        "pos": {k: [0, 0, 0, 0, 0] for k in POS_KEYS},   # [n, 1着, 2着, 3着, 着外]
        "kamashi": [0, 0], "tsuppari": [0, 0], "seri": [0, 0],
        "cut_head": [0, 0], "cut_mate": [0, 0],
    })
    for rid, cars in nar.items():
        r = res.get(rid)
        if not r:
            continue
        size: dict = defaultdict(int)
        for v in cars.values():
            size[v[0]] += 1
        has_kamashi = any(v[2] == "カマシ" for v in cars.values())
        for car, (li, pi, leg, sg) in cars.items():
            rr = r.get(car)
            name = nm.get((rid, car))
            if not rr or not name or rr[0] is None:
                continue          # 失格・欠車は着順が入らない
            pos, sb = rr
            a = acc[name]
            lab = _pos_label(pi, size[li])
            p = a["pos"][lab]
            p[0] += 1
            p[min(pos, 4)] += 1                     # 4着以降はまとめて着外
            if leg == "カマシ":
                a["kamashi"][1] += 1
                a["kamashi"][0] += int("B" in (sb or ""))
            if leg in ("先行", "押え先") and has_kamashi:
                a["tsuppari"][1] += 1
                a["tsuppari"][0] += int("B" in (sb or ""))
            if sg is not None:
                riv = [c2 for c2, v in cars.items()
                       if v[3] == sg and c2 != car and c2 in r and r[c2][0] is not None]
                if riv:
                    a["seri"][1] += 1
                    a["seri"][0] += int(all(pos < r[c2][0] for c2 in riv))
            # ライン分断（先頭視点 / 番手視点）。同じ事象を両側から数える
            if lab == "先頭":
                mate = [c2 for c2, v in cars.items() if v[0] == li and v[1] == 1]
                if mate and mate[0] in r and r[mate[0]][0] is not None:
                    a["cut_head"][1] += 1
                    a["cut_head"][0] += int(pos <= 3 and r[mate[0]][0] > 3)
            elif lab == "番手":
                head = [c2 for c2, v in cars.items() if v[0] == li and v[1] == 0]
                if head and head[0] in r and r[head[0]][0] is not None:
                    a["cut_mate"][1] += 1
                    a["cut_mate"][0] += int(pos > 3 and r[head[0]][0] <= 3)

    out: dict = {}
    for name, a in acc.items():
        pos = {}
        for k in POS_KEYS:
            n, w, s2, s3, _ = a["pos"][k]
            if not n:
                continue
            pos[k] = {"n": n, "win": _rate(w, n), "top2": _rate(w + s2, n),
                      "top3": _rate(w + s2 + s3, n),
                      "win_k": w, "top3_k": w + s2 + s3}
        one = lambda key: {"n": a[key][1], "k": a[key][0], "rate": _rate(*a[key])}
        out[name] = {"pos": pos, "kamashi": one("kamashi"), "tsuppari": one("tsuppari"),
                     "seri": one("seri"), "cut_head": one("cut_head"),
                     "cut_mate": one("cut_mate")}
    return out
