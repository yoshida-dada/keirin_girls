"""ギヤ倍数・バンクと展開（S/B取得・決まり手）の関係を実測する。

検証する2仮説:
  (1) ギヤ倍数が大きい選手ほど主導権(B)/初手先頭(S)を取りやすいか、決まり手が偏るか
  (2) バンク（333/400/500m・会場）で決まり手やS/B構造が変わるか

**単純な相関では判断しない。** 本プロジェクトでは「実在するが既存特徴に吸収済み」が
繰り返し出ているため、いずれも既知の強い予測子で層別した上での**増分**を見る:
  - ギヤ → b_count（主導権回数, 実測でB的中51.3%）の順位で層別してなお効くか
  - バンク → ペース区分（先行型人数の相対定義。現行の決まり手目安はこれだけで出している）
             で層別してなお効くか

  PYTHONIOENCODING=utf-8 python scripts/analyze_gear_bank.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.features import venue_meta as vm

KIM = ["逃", "捲", "差", "ク"]


def _load(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    gear = defaultdict(dict)      # rid -> car -> gear
    for rid, car, g in c.execute("SELECT race_id,car_number,gear_ratio FROM entries"):
        if g:
            gear[rid][car] = float(g)
    res = defaultdict(dict)       # rid -> car -> (pos, sb, kim)
    for rid, car, pos, sb, kim in c.execute(
            "SELECT race_id,car_number,position,sb,kimarite FROM results"):
        res[rid][car] = (pos, sb or "", kim or "")
    venue = {rid: v for rid, v in c.execute("SELECT race_id,venue_code FROM races")}
    bcnt = defaultdict(dict)
    for rid, car, b in c.execute("SELECT race_id,car_number,b_count FROM recent_form"):
        bcnt[rid][car] = (b or 0)
    c.close()
    return gear, res, venue, bcnt


def _pace(bv: dict[int, float]) -> str:
    vals = list(bv.values())
    mx = max(vals) if vals else 0
    if mx < 2:
        return "スロー"
    n = sum(1 for v in vals if v >= 0.4 * mx)
    return "スロー" if n <= 2 else ("ハイ" if n >= 4 else "ミドル")


def _rate(rows, key):
    n = len(rows)
    return (sum(1 for r in rows if r[key]) / n * 100) if n else 0.0


def _kim_dist(rows) -> str:
    n = len(rows) or 1
    d = {k: sum(1 for r in rows if r["kim"] == k) / n * 100 for k in KIM}
    return "  ".join(f"{k}{d[k]:>5.1f}%" for k in KIM)


def _cell(wins: list, brows: list) -> dict:
    """決まり手構成＋主導権(B)取得者の連対/着外率を1セル分にまとめる。"""
    n = len(wins)
    d = {"n": n, "kim": {k: round(sum(1 for r in wins if r["kim"] == k) / n * 100, 1)
                         for k in ("逃", "捲", "差")} if n else {}}
    if brows:
        d["b_n"] = len(brows)
        d["b_rentai"] = round(sum(1 for r in brows if r["bpos"] and r["bpos"] <= 2) / len(brows) * 100)
        d["b_gaiji"] = round(sum(1 for r in brows if r["bpos"] and r["bpos"] >= 4) / len(brows) * 100)
    return d


def emit_stats(wins: list, brows: list, out: Path, n_races: int) -> None:
    """本番（DB非依存）から引く決まり手テーブルを書き出す。

    参照順: (ペース×バンク) → バンク → ペース → 全体。薄いセルは上位へフォールバックさせる。
    """
    import json
    from datetime import date

    def sel(rows, pace=None, bank=None):
        return [r for r in rows if (pace is None or r["pace"] == pace)
                and (bank is None or r["bank"] == bank)]

    doc = {"generated": str(date.today()), "n_races": n_races, "min_n": 60,
           "cells": {}, "bank": {}, "pace": {},
           "global": _cell(wins, brows),
           "note": "勝ち決まり手の構成比(%)と主導権(B)取得者の連対/着外率。"
                   "バンク長は決まり手への影響がペースより大きい（逃げ率 333m 35.3% vs 500m 12.8%）。"}
    for pace in ("スロー", "ミドル", "ハイ"):
        doc["pace"][pace] = _cell(sel(wins, pace=pace), sel(brows, pace=pace))
        for bank in (333, 400, 500):
            doc["cells"][f"{pace}|{bank}"] = _cell(sel(wins, pace, bank), sel(brows, pace, bank))
    for bank in (333, 400, 500):
        doc["bank"][str(bank)] = _cell(sel(wins, bank=bank), sel(brows, bank=bank))
    # 会場別（表示用の参考値）。標本が薄い（n=67〜292）ため予測の重みには使わない
    # ＝会場別重みは検証で不採用（scripts/validate_venue_interaction.py）。表示のみ。
    doc["venue"] = {}
    for v in sorted({r["venue_code"] for r in wins if r.get("venue_code")}):
        w = [r for r in wins if r.get("venue_code") == v]
        if len(w) >= 40:
            doc["venue"][v] = _cell(w, [r for r in brows if r.get("venue_code") == v])
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n統計を書き出し: {out}  (勝者{len(wins)} / B取得{len(brows)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="ギヤ・バンクと展開の関係")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--emit", help="本番参照用の決まり手テーブルJSONの書き出し先")
    args = ap.parse_args()
    gear, res, venue, bcnt = _load(args.db)

    rows = []          # 1出走 = 1行
    wins = []          # 勝者のみ
    brows = []         # 主導権(B)取得者のみ（着順つき＝主導権の信頼度用）
    for rid, gmap in gear.items():
        rmap = res.get(rid)
        if not rmap or len(gmap) < 5:
            continue
        cars = [c for c in gmap if c in rmap]
        if len(cars) < 5:
            continue
        gs = [gmap[c] for c in cars]
        gmean = sum(gs) / len(gs)
        grank = {c: i for i, c in enumerate(sorted(cars, key=lambda x: -gmap[x]))}   # 0=最大ギヤ
        bv = {c: bcnt.get(rid, {}).get(c, 0) for c in cars}
        brank = {c: i for i, c in enumerate(sorted(cars, key=lambda x: -bv[x]))}     # 0=B回数最多
        pace = _pace(bv)
        bank = vm.bank_length(venue.get(rid, ""))
        for c in cars:
            pos, sb, kim = rmap[c]
            r = {"rid": rid, "car": c, "gear": gmap[c], "grel": gmap[c] - gmean,
                 "grank": grank[c], "brank": brank[c], "bank": bank, "pace": pace,
                 "venue_code": venue.get(rid),
                 "S": "S" in sb, "B": "B" in sb, "win": pos == 1, "kim": kim,
                 "uniq_gear": len(set(gs)) > 1}
            rows.append(r)
            if pos == 1:
                wins.append(r)
            if r["B"]:
                brows.append({**r, "bpos": pos})

    print(f"対象 {len(set(r['rid'] for r in rows))}レース / {len(rows)}出走\n")

    # ---- ギヤ倍数そのものの分布（ガールズは上限規制がある） ----
    gv = sorted(set(round(r["gear"], 2) for r in rows))
    print(f"【ギヤ倍数の値域】{min(gv)}〜{max(gv)}  種類{len(gv)}")
    cnt = defaultdict(int)
    for r in rows:
        cnt[round(r["gear"], 2)] += 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:8]
    print("  頻度上位: " + "  ".join(f"{g}:{n}({n/len(rows):.0%})" for g, n in top))
    same = sum(1 for r in rows if not r["uniq_gear"]) / len(rows) * 100
    print(f"  レース内で全員同一ギヤ: {same:.1f}% の出走が該当\n")

    # ---- (1) ギヤ順位 × S/B/勝率 ----
    print("【ギヤのレース内順位 × 展開】0=そのレースで最大ギヤ")
    print(f"  {'順位':<6}{'n':>7}{'S取得':>8}{'B取得':>8}{'勝率':>8}   勝者の決まり手")
    for k in range(7):
        sub = [r for r in rows if r["grank"] == k and r["uniq_gear"]]
        if len(sub) < 100:
            continue
        w = [r for r in sub if r["win"]]
        print(f"  {k:<6}{len(sub):>7}{_rate(sub,'S'):>7.1f}%{_rate(sub,'B'):>7.1f}%"
              f"{_rate(sub,'win'):>7.1f}%   {_kim_dist(w)}")

    # ---- (1b) 増分: b_count順位で層別してもギヤが効くか ----
    print("\n【増分確認】b_count順位で層別 → その中でギヤ上位/下位のB取得率")
    print(f"  {'b_count順位':<12}{'ギヤ上位半n':>12}{'B取得':>8}{'ギヤ下位半n':>12}{'B取得':>8}{'差':>8}")
    for bk in range(4):
        sub = [r for r in rows if r["brank"] == bk and r["uniq_gear"]]
        if len(sub) < 200:
            continue
        hi = [r for r in sub if r["grank"] <= 2]
        lo = [r for r in sub if r["grank"] >= 4]
        if len(hi) < 50 or len(lo) < 50:
            continue
        d = _rate(hi, 'B') - _rate(lo, 'B')
        print(f"  {bk:<12}{len(hi):>12}{_rate(hi,'B'):>7.1f}%{len(lo):>12}{_rate(lo,'B'):>7.1f}%{d:>+7.1f}")

    # ---- (2) バンク × 決まり手 ----
    print("\n【バンク長 × 勝者の決まり手】")
    print(f"  {'バンク':<8}{'n':>7}   決まり手")
    for bank in (333, 400, 500):
        w = [r for r in wins if r["bank"] == bank]
        if len(w) < 100:
            continue
        print(f"  {str(bank)+'m':<8}{len(w):>7}   {_kim_dist(w)}")

    # ---- (2b) 増分: ペース区分で層別してもバンクが効くか ----
    print("\n【増分確認】ペース区分で層別 → その中でバンク別の決まり手")
    for pace in ("スロー", "ミドル", "ハイ"):
        print(f"  [{pace}]")
        base = [r for r in wins if r["pace"] == pace]
        if base:
            print(f"    {'全体':<8}{len(base):>7}   {_kim_dist(base)}")
        for bank in (333, 400, 500):
            w = [r for r in base if r["bank"] == bank]
            if len(w) < 60:
                continue
            print(f"    {str(bank)+'m':<8}{len(w):>7}   {_kim_dist(w)}")

    # ---- (2c) 会場ごとのばらつき（バンク長だけでは説明できない差があるか） ----
    print("\n【会場別の勝者決まり手】n>=150 のみ・逃げ率順")
    byv = defaultdict(list)
    for r in wins:
        byv[venue.get(r["rid"], "")].append(r)
    items = [(v, w) for v, w in byv.items() if len(w) >= 150]
    items.sort(key=lambda kv: -sum(1 for r in kv[1] if r["kim"] == "逃") / len(kv[1]))
    for v, w in items:
        bank = vm.bank_length(v)
        print(f"  {v}({bank}m) n={len(w):>5}   {_kim_dist(w)}")

    if args.emit:
        emit_stats(wins, brows, Path(args.emit), len(set(r["rid"] for r in rows)))


if __name__ == "__main__":
    main()
